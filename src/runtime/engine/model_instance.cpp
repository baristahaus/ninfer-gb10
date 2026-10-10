#include <iostream>
#include <set>
#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/binder.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/registry.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// The kernel's reclaimable view of host memory (/proc/meminfo MemAvailable) in bytes.
std::size_t host_available_bytes() {
    std::ifstream info("/proc/meminfo");
    std::string key;
    std::string unit;
    std::size_t kib = 0;
    while (info >> key >> kib >> unit) {
        if (key == "MemAvailable:") { return kib * 1024; }
    }
    throw std::runtime_error("/proc/meminfo reports no MemAvailable");
}

// Host memory an integrated device keeps out of Device sizing beyond the pinned Host KV arena:
// Host StateImages, the resident part of the file-mapped PLE table, and the process itself. On
// GB10 the eight default StateImages take about 1 GiB and a served run keeps about 3 GiB of PLE
// pages cached.
constexpr std::size_t kIntegratedHostReserveBytes = 6ULL << 30;

// Device bytes a new allocation can still obtain, before the Program's pinned Host KV arena is
// allocated when `host_kv_pending` is set.
//
// A discrete device reports its own memory through cudaMemGetInfo. An integrated device (GB10)
// shares host memory, and cudaMalloc reclaims clean page cache that cudaMemGetInfo counts as used
// (measured: 104 GiB obtained against 10 GiB reported free with the artifact cached), so its
// budget is MemAvailable less the host memory that must stay out of Device use.
std::size_t obtainable_device_bytes(const DeviceContext& device, const EngineOptions& options,
                                    bool host_kv_pending) {
    if (!device.props.integrated) { return current_free_device_bytes(); }
    const std::size_t reserve =
        kIntegratedHostReserveBytes +
        (host_kv_pending ? options.context_cache.host_kv_capacity_bytes : 0);
    const std::size_t available = host_available_bytes();
    return available > reserve ? available - reserve : 0;
}

// An integrated device shares host memory with every process. A process that just exited (a
// restarted server) returns its tens of GiB over several seconds, so sizing from MemAvailable at
// that moment fails the weights check or silently plans a smaller KV cache. Wait until
// MemAvailable stops rising before any sizing: stable means under 256 MiB of growth across one
// 500 ms window. Bounded so a busy host never blocks startup indefinitely.
void wait_for_integrated_memory_release(const DeviceContext& device) {
    if (!device.props.integrated) { return; }
    constexpr auto kWindow           = std::chrono::milliseconds(500);
    constexpr auto kLimit            = std::chrono::seconds(60);
    constexpr std::size_t kStableGap = 256ULL << 20;
    const auto started               = Clock::now();
    const std::size_t initial        = host_available_bytes();
    std::size_t previous             = initial;
    for (;;) {
        std::this_thread::sleep_for(kWindow);
        const std::size_t current = host_available_bytes();
        const bool stable         = current < previous + kStableGap;
        const bool expired        = Clock::now() - started >= kLimit;
        if (stable || expired) {
            if (current > initial + kStableGap) {
                std::fprintf(stderr,
                             "ninfer: waited %.1f s for host memory release (%.1f -> %.1f GiB "
                             "available)%s\n",
                             std::chrono::duration<double>(Clock::now() - started).count(),
                             static_cast<double>(initial) / (1ULL << 30),
                             static_cast<double>(current) / (1ULL << 30),
                             stable ? "" : "; still rising at the limit");
            }
            return;
        }
        previous = current;
    }
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    if (!options.capture_path.empty()) {
        auto ancestor = std::filesystem::weakly_canonical(options.capture_path);
        while (!ancestor.empty()) {
            if (ancestor == std::filesystem::weakly_canonical(options.artifact_path).parent_path() ||
                std::filesystem::exists(ancestor / "config.json") ||
                std::filesystem::exists(ancestor / "model.safetensors.index.json"))
                throw std::invalid_argument("--capture-path must not be inside a model directory");
            const auto parent = ancestor.parent_path();
            if (parent == ancestor) break;
            ancestor = parent;
        }
        std::clog << "--capture-path: capture requests bypass prefix reuse\n";
        if (options.capture_sites.empty()) throw std::invalid_argument("capture-sites must be nonempty");
        std::set<std::string> unique;
        for (const auto& site : options.capture_sites)
            if ((site != "prompt_last" && site != "completion_last") || !unique.insert(site).second)
                throw std::invalid_argument("invalid or duplicate capture site: " + site);
    }
    if (!options.capture_path.empty() && options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::Mtp)
        throw std::invalid_argument("capture supports ordinary and MTP decoding");
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_flash_next(const EngineOptions& options, DeviceContext& device,
                                      artifact::Reader& reader, Clock::time_point start) {
    using Model        = FlashNextInstance::ModelContract;
    const auto& config = reader.directory().component("text").config;
    if (config.at("model_type") != "qwen3_8_flash_next_text" || config.at("hidden_size") != 2560 ||
        config.at("num_hidden_layers") != 48 || config.at("vocab_size") != 248320 ||
        config.at("num_experts") != 512) {
        throw artifact::ArtifactError("unsupported Flash-Next configuration");
    }
    if (!options.chat_template_path.empty()) {
        throw std::invalid_argument("Flash-Next uses its registered artifact chat template; "
                                    "--chat-template is unsupported");
    }
    artifact::Binder binder(reader);
    const auto profile = Model::weights_profile(reader);
    auto plan        = Model::plan_load(binder, options, profile);
    auto planner     = Model::make_sequence_planner(device, options, profile);
    const auto curve = planner.capacity_curve();
    const auto free  = obtainable_device_bytes(device, options, true);
    if (plan.materialization().device_capacity_bytes > free) {
        throw std::invalid_argument("Flash-Next weights exceed free GPU memory");
    }
    (void)resolve_kv_capacity(options.kv_capacity, curve,
                              free - plan.materialization().device_capacity_bytes);
    std::set<std::string> formats{
        profile == Model::WeightsProfile::Nvfp4Bf16Ple ? "bf16" : "fp8_e4m3fn"};
    for (const auto& placement : plan.materialization().device_objects) {
        formats.insert(reader.directory().tensor(placement.object).format);
    }
    auto backing     = artifact::materialize(reader, std::move(plan.materialization()), device,
                                             &options.startup_observer);
    const auto stats = backing.stats();
    auto model       = Model::construct_loaded_model(std::move(plan), std::move(backing));
    auto instance    = std::make_unique<FlashNextInstance>(std::move(model), options);
    auto resolution  = resolve_kv_capacity(options.kv_capacity, curve,
                                           obtainable_device_bytes(device, options, true));
    auto sequence    = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("Flash-Next resolved KV capacity differs from finalized plan");
    }
    instance->kv_capacity_resolution = resolution;
    instance->program = Model::create_program(*instance->model, std::move(sequence), device,
                                              options.startup_observer);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        obtainable_device_bytes(device, options, false);
    auto cost                                                      = resolve_context_machine_cost(
        {.hardware_class =
             context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
         .prefill_signature = "qwen3_8_flash_next_125b_a6b-nvfp4-v3-1"},
        options.context_cost.preset_path);
    LoadSummary summary;
    summary.architecture = "Qwen3_8FlashNextForCausalLM";
    summary.model_name   = reader.directory().metadata.value("name", "qwen3.8-flash-next-125b-a6b");
    summary.prefill_signature = "qwen3_8_flash_next_125b_a6b-nvfp4-v3-1";
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(cost.summary);
    return {std::move(instance), std::move(summary), std::move(cost.model)};
}

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    wait_for_integrated_memory_release(device);
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    const auto& config        = reader.directory().component("text").config;
    const auto& architectures = config.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1) {
        throw artifact::ArtifactError("text component must select one architecture");
    }
    const auto architecture = models::resolve_architecture(
        architectures.front().get<std::string>(), config.at("model_type").get<std::string>());
    if (architecture == models::Architecture::Qwen3_8FlashNext) {
        return construct_flash_next(options, device, reader, start);
    }
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
             context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          obtainable_device_bytes(device, options, true));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        obtainable_device_bytes(device, options, false);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
