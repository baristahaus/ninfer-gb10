#include "artifact/identity.h"
#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include <cmath>
#include <iostream>
#include <mutex>
#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.execution.capture                 = options.execution.capture;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    template <class Prompt>
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         Prompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    std::variant<models::qwen3_5::PreparedPrompt, models::qwen3_8_flash_next::PreparedPrompt> value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling, SteeringState steering = {})
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          steering_(std::move(steering)), sampling_(sampling) {}

    const SteeringState& resolved_steering() const noexcept { return steering_; }
    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    SteeringState steering_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

const SteeringState& GenerationHandle::resolved_steering() const noexcept {
    static const SteeringState empty;
    return impl_ ? impl_->resolved_steering() : empty;
}

class Engine::Impl {
public:
    using GenerationCore      = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore         = runtime::CausalScoreCore<runtime::ModelInstance>;
    using FlashGenerationCore = runtime::EngineCore<runtime::FlashNextInstance>;
    using FlashScoringCore    = runtime::CausalScoreCore<runtime::FlashNextInstance>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>,
                     std::unique_ptr<FlashGenerationCore>, std::unique_ptr<FlashScoringCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        std::optional<models::qwen3_8_flash_next::SteeringPack> startup_pack;
        if (!options.steering_pack.empty()) startup_pack = models::qwen3_8_flash_next::read_steering_pack(options.steering_pack);
        auto constructed = runtime::construct_model(options, device);
        active           = std::move(constructed.instance);
        load             = std::move(constructed.load);
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        std::visit(
            [&](auto& instance) {
                using Instance = typename std::remove_cvref_t<decltype(instance)>::element_type;
                if constexpr (std::is_same_v<Instance, runtime::FlashNextInstance>) {
                    sampling_defaults = Instance::ModelContract::sampling_defaults(
                        Instance::ModelContract::model_id);
                } else {
                    sampling_defaults = instance->frontend.sampling_defaults();
                }
                if constexpr (std::is_same_v<Instance, runtime::FlashNextInstance>) {
                    if (!options.capture_path.empty() || startup_pack) {
                        std::clog << "Computing native encoded-object SHA256 identities serially (source digests cannot be recovered after conversion)\n";
                        identity = artifact::encoded_identity(options.artifact_path, options.chat_template_path);
                    }
                    instance->program->configure_activation(options, identity.is_null() ? "" : identity.dump());
                    if (startup_pack) {
                        instance->program->activate_steering(&*startup_pack);
                        steering = startup_pack->state;
                        steering.generation = 1;
                        log_pack(&*startup_pack, options.steering_pack);
                    }
                } else if (!options.capture_path.empty() || startup_pack) {
                    throw std::invalid_argument("capture and steering require Qwen3.8 Flash-Next");
                }
                if (options.purpose == EnginePurpose::CausalScoring) {
                    core = std::make_unique<runtime::CausalScoreCore<Instance>>(*instance, device);
                } else {
                    core = std::make_unique<runtime::EngineCore<Instance>>(
                        *instance, device, options, std::move(constructed.context_cost));
                }
            },
            active);
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    void log_pack(const models::qwen3_8_flash_next::SteeringPack* pack, const std::filesystem::path& path) const {
        nlohmann::json record{{"event","steering_activation"},{"pack",path.string()},
            {"pack_sha",steering.pack_sha},{"rank",steering.rank},{"layers",steering.layers},
            {"generation",steering.generation},{"default_steering_strength",steering.strength},
            {"norm_preserve",pack ? pack->norm_preserve : false},{"served_identity",identity}};
        if (pack) {
            record["pack_identity"] = pack->metadata;
            for (const char* field : {"model_sha","config_sha","template_sha"})
                if (identity.value(field,"") != pack->metadata.value(field,""))
                    record["identity_warning"] = "pack and served artifact identities differ (recorded, not enforced)";
        }
        std::clog << record.dump() << '\n';
    }
    std::mutex activation_mutex;
    SteeringState steering;
    nlohmann::json identity;
    EngineOptions options;
    DeviceContext device;
    runtime::ActiveModel active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& instance) -> PreparedPrompt {
            auto prepared      = instance->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            const SamplingMode sampling_mode =
                info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
            if (info.prompt_tokens > instance->capacity) {
                throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& instance) -> PreparedPrompt {
            if (token_ids.size() > instance->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   context_capacity_error(token_ids.size(), instance->capacity));
            }
            auto prepared =
                instance->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > instance->capacity) {
                throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit([&](const auto& instance) { return instance->frontend.tokenize_text(text); },
                      impl_->active);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr ((std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>> ||
                           std::is_same_v<CoreState, std::unique_ptr<Impl::FlashScoringCore>>)) {
                using Prompt = typename CoreState::element_type::PreparedPrompt;
                auto* value  = std::get_if<Prompt>(&prompt.impl_->value);
                if (!value) {
                    throw std::invalid_argument("prompt belongs to another model family");
                }
                return core->score(std::move(*value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& instance) {
            return instance->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::lock_guard activation_lock(impl_->activation_mutex);
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    SteeringState steering = impl_->steering;
    steering.strength = options.execution.steering_strength.value_or(steering.strength);
    if (!std::isfinite(steering.strength) || steering.strength < 0 || steering.strength > 1)
        throw std::invalid_argument("steering strength must be in [0,1]");
    if (steering.strength != 0 && impl_->steering.rank == 0)
        throw std::invalid_argument("nonzero steering strength requires an active pack");
    if (options.execution.capture) {
        if (impl_->options.capture_path.empty()) throw std::invalid_argument("capture requires --capture-path");
        if (options.execution.requested_output_tokens == 0)
            throw std::invalid_argument("capture requires at least one generated token");
    }
    if (options.execution.capture) options.execution.allow_prefix_reuse = false;
    std::visit([&](auto& value) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, models::qwen3_8_flash_next::PreparedPrompt>)
            models::qwen3_8_flash_next::PreparedPromptAccess::set_steering(value, steering.generation, steering.strength);
        else if (options.execution.capture || steering.strength != 0)
            throw std::invalid_argument("capture and steering require Qwen3.8 Flash-Next");
    }, prompt.impl_->value);
    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    resolved_options.execution.steering = steering;
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.steering                   = steering;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling, steering));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr ((std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>> ||
                                  std::is_same_v<CoreState,
                                                 std::unique_ptr<Impl::FlashScoringCore>>)) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                using Prompt = typename CoreState::element_type::PreparedPrompt;
                auto* value  = std::get_if<Prompt>(&prompt.impl_->value);
                if (!value) {
                    throw std::invalid_argument("prompt belongs to another model family");
                }
                auto submission = core->submit(std::move(*value), prompt_summary, prepare_seconds,
                                               std::move(resolved_options), consumer_mode,
                                               observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling, steering));
            }
        },
        impl_->core);
}

SteeringState Engine::activate_steering(const std::optional<std::filesystem::path>& path) {
    if (!impl_) throw std::logic_error("Engine is moved from");
    std::optional<models::qwen3_8_flash_next::SteeringPack> pack;
    if (path) pack = models::qwen3_8_flash_next::read_steering_pack(*path);
    std::lock_guard lock(impl_->activation_mutex);
    auto* core = std::get_if<std::unique_ptr<Impl::FlashGenerationCore>>(&impl_->core);
    if (!core) throw std::invalid_argument("steering requires a Flash-Next generation Engine");
    if (impl_->identity.is_null()) {
        std::clog << "Computing native encoded-object SHA256 identities serially (source digests cannot be recovered after conversion)\n";
        impl_->identity = artifact::encoded_identity(impl_->options.artifact_path, impl_->options.chat_template_path);
    }
    (*core)->update_idle_program([&](auto& program) {
        program.activate_steering(pack ? &*pack : nullptr);
    });
    const auto generation = impl_->steering.generation + 1;
    impl_->steering = pack ? pack->state : SteeringState{};
    impl_->steering.generation = generation;
    impl_->log_pack(pack ? &*pack : nullptr, path.value_or(std::filesystem::path{}));
    return impl_->steering;
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& instance) { return instance->frontend.media_cache_summary(); },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
