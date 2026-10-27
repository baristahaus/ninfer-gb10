#include "runtime/engine/flash_next/startup.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/registry.h"
#include "runtime/engine/context_cache/context_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::flashnext_runtime {
namespace {
using Clock = std::chrono::steady_clock;

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
// Host StateImages, the resident part of the file-mapped PLE table, and the process itself.
constexpr std::size_t kIntegratedHostReserveBytes = 6ULL << 30;

// Device bytes a new allocation can still obtain, before the Program's pinned Host KV arena is
// allocated when `host_kv_pending` is set.
//
// A discrete device reports its own memory through cudaMemGetInfo. An integrated device shares
// host memory, and cudaMalloc reclaims clean page cache that cudaMemGetInfo counts as used, so
// its budget is MemAvailable less the host memory that must stay out of Device use.
std::size_t obtainable_device_bytes(const DeviceContext& device, const EngineOptions& options,
                                    bool host_kv_pending) {
    if (!device.properties().integrated) { return current_free_device_bytes(); }
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
    if (!device.properties().integrated) { return; }
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

// Flash-Next cache sizing follows the master-generation contract: Device StateImage slots default
// to the concurrency, private continuations to twice the concurrency, shared prefixes to the
// larger of concurrency and the marker cap, and two long anchors per continuation. A disabled
// cache accepts only root-only capacities.
void normalize_flash_next_cache_options(EngineOptions& options) {
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
        return;
    }
    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes =
        cache.max_shared_prefixes.value_or(std::max(concurrency, static_cast<std::uint32_t>(
                                                                   kMaximumExplicitPromptCacheMarkers)));
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
}

ConstructedFlashNextModel construct_flash_next(EngineOptions& options, artifact::Reader& reader,
                                               Clock::time_point start) {
    using Model = FlashNextInstance::ModelContract;
    StartupPhaseScope device_phase(options.startup_observer, StartupPhase::CudaInitialize);
    auto device = std::make_unique<DeviceContext>(options.device);
    device_phase.complete();
    wait_for_integrated_memory_release(*device);

    if (options.prefill_chunk == 0) {
        options.prefill_chunk = device->properties().integrated ? kIntegratedPrefillChunk
                                                                : kDiscretePrefillChunk;
    }
    normalize_flash_next_cache_options(options);

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
    auto plan        = Model::plan_load(binder, options, Model::WeightsProfile::Nvfp4);
    auto planner     = Model::make_sequence_planner(*device, options, Model::WeightsProfile::Nvfp4);
    const auto curve = planner.capacity_curve();

    std::vector<artifact::PlacementRequest> requests;
    requests.reserve(plan.device_objects().size());
    for (const auto object : plan.device_objects()) {
        requests.push_back({object, device->device(), artifact::HostAccess::CpuOnly});
    }
    auto storage =
        artifact::plan_materialization(reader, requests, std::move(plan.materialization()));

    const auto free = obtainable_device_bytes(*device, options, true);
    if (storage.device_capacity_bytes > free) {
        throw std::invalid_argument("Flash-Next weights exceed free GPU memory");
    }
    (void)resolve_kv_capacity(options.kv_capacity, curve, free - storage.device_capacity_bytes);

    std::set<std::string> formats{"fp8_e4m3fn"};
    for (const auto& request : requests) {
        formats.insert(reader.directory().tensor(request.object).format);
    }

    StartupPhaseScope materialize_phase(options.startup_observer, StartupPhase::WeightsMaterialize);
    auto backing =
        artifact::materialize(reader, std::move(storage), device->transfer_view(),
                              &options.startup_observer);
    materialize_phase.complete();
    const auto stats = backing.stats();
    auto model       = Model::construct_loaded_model(std::move(plan), std::move(backing));
    auto instance    = std::make_unique<FlashNextInstance>(std::move(model), options);

    StartupPhaseScope planning_phase(options.startup_observer, StartupPhase::TargetFinalize);
    const auto budget = obtainable_device_bytes(*device, options, true);
    auto resolution   = resolve_kv_capacity(options.kv_capacity, curve, budget);
    auto sequence     = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("Flash-Next resolved KV capacity differs from finalized plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning_phase.complete();

    StartupPhaseScope program_phase(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = Model::create_program(*instance->model, std::move(sequence), *device,
                                              options.startup_observer);
    device->synchronize();
    program_phase.complete();

    instance->kv_capacity_resolution.available_after_startup_bytes =
        obtainable_device_bytes(*device, options, false);

    auto cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(device->properties().name,
                                                       device->properties().major,
                                                       device->properties().minor),
         .prefill_signature = "qwen3_8_flash_next_125b_a6b-nvfp4-v3-1"},
        options.context_cost.preset_path);
    LoadSummary summary;
    summary.architecture = "Qwen3_8FlashNextForCausalLM";
    summary.model_name   = reader.directory().metadata.value("name", "qwen3.8-flash-next-125b-a6b");
    summary.prefill_signature = "qwen3_8_flash_next_125b_a6b-nvfp4-v3-1";
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds             = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds           = stats.upload_seconds;
    summary.artifact_bytes_read      = stats.read_bytes;
    summary.host_to_device_bytes     = stats.h2d_bytes;
    summary.max_rank_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count      = stats.device_object_count;
    summary.host_object_count        = stats.host_object_count;
    summary.context_cost             = std::move(cost.summary);

    ConstructedFlashNextModel constructed;
    constructed.instance     = std::move(instance);
    constructed.load         = std::move(summary);
    constructed.context_cost = std::move(cost.model);
    constructed.device       = std::move(device);
    return constructed;
}

} // namespace ninfer::flashnext_runtime
