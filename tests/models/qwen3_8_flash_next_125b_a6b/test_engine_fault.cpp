// End-to-end check of the Engine worker's failure policy (contract 7.4), driven through
// the test-only NINFER_ENGINE_FAULT_INJECTION seam:
//
// - A std::bad_alloc at a worker boundary recovers: the affected admitted request ends
//   with a retryable Overloaded error, the pending FIFO still completes, and the Engine
//   keeps serving.
// - A std::logic_error (invariant violation) fails the Engine as a whole: every
//   admitted and pending request fails, and new requests are rejected as Unavailable.
//
// The real artifact is required (NINFER_QWEN38_FLASH_NEXT_WEIGHTS); the test skips with
// 77 when it is absent.
#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens         = 0;
    options.enable_vision                    = false;
    options.use_cuda_graph                   = true;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 4;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes       = 1;
    return options;
}

const std::vector<ninfer::TokenId>& canonical_prompt() {
    static const std::vector<ninfer::TokenId> prompt{
        248045, 846, 198,  814, 20139,  303, 2250,   2716,  22157, 3069,   279, 12515,  7701, 6105,
        2261,   279, 1834, 13,  248046, 198, 248045, 74455, 198,   248068, 271, 248069, 271};
    return prompt;
}

// A long prompt: the OOM fault fires at the next worker boundary after it is armed, which
// is during the first request's prefill. A long prefill makes it impossible for the second
// request to be admitted before the fault, so the recovery always finds it pending.
std::vector<ninfer::TokenId> long_prompt() {
    std::vector<ninfer::TokenId> prompt = canonical_prompt();
    for (int index = 0; index < 160; ++index) { prompt.push_back(20139); }
    return prompt;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

// Arms the fault when the streaming consumer receives the generation-start event. That
// event is published when the request is admitted, so the armed fault fires at the next
// worker boundary with the request already admitted (materializing or active).
class ArmOnStartSink final : public ninfer::OutputSink {
public:
    explicit ArmOnStartSink(ninfer::Engine& engine) : engine_(engine) {}

    void start(ninfer::GenerationStart) override {
        engine_.arm_next_worker_fault(ninfer::Engine::WorkerFault::Oom);
    }

    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta) override {}

private:
    ninfer::Engine& engine_;
};

// OOM recovery: the admitted request ends with a retryable Overloaded error, the pending
// request still completes, and the Engine keeps serving.
int exercise_oom_recovery(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    ArmOnStartSink sink(engine);
    auto first = engine.submit(engine.prepare_tokens(long_prompt()), greedy_options(16),
                               ninfer::OutputConsumerMode::Streaming);
    std::exception_ptr first_error;
    std::thread first_consumer([&] {
        try {
            first.wait(&sink);
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    auto second = engine.submit(engine.prepare_tokens(long_prompt()), greedy_options(16));

    const bool second_completed = [&] {
        try {
            const auto result = second.wait();
            return result.generated_token_ids.size() == 16 &&
                   result.finish_reason == ninfer::FinishReason::OutputLimit;
        } catch (...) {
            return false;
        }
    }();
    first_consumer.join();

    const bool first_overloaded = [&] {
        try {
            std::rethrow_exception(first_error);
            return false;
        } catch (const ninfer::RequestError& error) {
            return error.kind() == ninfer::RequestErrorKind::Overloaded;
        } catch (...) {
            return false;
        }
    }();

    if (!first_overloaded || !second_completed) {
        std::cerr << "Flash-Next OOM recovery: first_overloaded=" << first_overloaded
                  << " second_completed=" << second_completed << '\n';
        return 1;
    }
    const bool third_completed = [&] {
        try {
            const auto result =
                engine.generate(engine.prepare_tokens(canonical_prompt()), greedy_options(8));
            return result.generated_token_ids.size() == 8;
        } catch (...) {
            return false;
        }
    }();
    if (!third_completed) {
        std::cerr << "Flash-Next OOM recovery: the Engine is not serving after recovery\n";
        return 1;
    }
    return 0;
}

// Invariant violation: the Engine fails as a whole; every admitted and pending request
// fails, and new requests are rejected.
int exercise_invariant_failure(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    auto first  = engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(16));
    auto second = engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(16));
    engine.arm_next_worker_fault(ninfer::Engine::WorkerFault::InvariantError);

    std::exception_ptr first_error;
    std::thread first_consumer([&] {
        try {
            first.wait();
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    std::exception_ptr second_error;
    try {
        second.wait();
    } catch (...) {
        second_error = std::current_exception();
    }
    first_consumer.join();

    bool submission_unavailable = false;
    try {
        (void)engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(8));
    } catch (const ninfer::RequestError& error) {
        submission_unavailable = error.kind() == ninfer::RequestErrorKind::Unavailable;
    }

    if (first_error == nullptr || second_error == nullptr || !submission_unavailable ||
        engine.is_available()) {
        std::cerr << "Flash-Next invariant failure: first_failed="
                  << (first_error != nullptr) << " second_failed="
                  << (second_error != nullptr) << " submission_unavailable="
                  << submission_unavailable << " is_available=" << engine.is_available() << '\n';
        return 1;
    }
    return 0;
}

// Consecutive-recovery cap: nine queued OOMs, none followed by a successful work unit.
// The first eight recover (the request stays pending: every unit rethrows before
// admission); the ninth trips the cap and fails the Engine.
int exercise_oom_recovery_cap(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    for (int index = 0; index < 9; ++index) {
        engine.arm_next_worker_fault(ninfer::Engine::WorkerFault::Oom);
    }
    auto handle = engine.submit(engine.prepare_tokens(long_prompt()), greedy_options(16));
    bool request_failed = false;
    try {
        handle.wait();
    } catch (...) {
        request_failed = true;
    }

    bool submission_unavailable = false;
    try {
        (void)engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(8));
    } catch (const ninfer::RequestError& error) {
        submission_unavailable = error.kind() == ninfer::RequestErrorKind::Unavailable;
    }
    if (!request_failed || engine.is_available() || !submission_unavailable) {
        std::cerr << "Flash-Next OOM cap: request_failed=" << request_failed
                  << " is_available=" << engine.is_available()
                  << " submission_unavailable=" << submission_unavailable << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
#if !defined(NINFER_ENGINE_FAULT_INJECTION)
    std::cerr << "test built without NINFER_ENGINE_FAULT_INJECTION\n";
    return 1;
#else
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        if (exercise_oom_recovery(artifact) != 0) { return 1; }
        if (exercise_oom_recovery_cap(artifact) != 0) { return 1; }
        if (exercise_invariant_failure(artifact) != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next fault policy: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK Qwen3.8 Flash Next fault policy\n";
    return 0;
#endif
}
