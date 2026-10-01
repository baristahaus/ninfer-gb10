// F1 oracle-shape probe (2026-10-02). Runs the real test's canonical prompt +
// golden through the public Engine API at several pool shapes (page size
// 64/max_concurrency) and reports which shapes reproduce the committed
// oracle output. Mirrors exercise_ordinary_greedy (plain decode, greedy,
// 12 tokens); only max_concurrency/max_context/kv_capacity vary.
// Shapes:
//   C1  probe shape: ctx 73728, kv 73728, conc 1  (page 64, pool 1152)
//   C4  probe shape: ctx 73728, kv 73728, conc 4  (page 16, pool 4608)
//   T2  real-test shape: ctx 512, kv 1024, conc 2 (page 32, pool 32)
//   C4s small C4 shape: ctx 512, kv 1024, conc 4  (page 16, pool 64)
//   C8s page-8 shape:   ctx 512, kv 1024, conc 8  (page 8, pool 128)
#include <cstdint>
#include <iostream>
#include <vector>

#include "ninfer/engine.h"
#include "ninfer/types.h"

static const std::vector<ninfer::TokenId> kPrompt{
    248045, 846,  198,  814,   20139, 303,  2250,  2716, 22157, 3069, 279,
    12515,  7701,  6105,  2261,  279,   1834,  248046, 198,  248045, 74455,
    198,    248068, 271,  248069, 271};

// canonical_output() in test_engine_real.cpp
static const std::vector<ninfer::TokenId> kGolden{
    29108, 4009, 27891, 8964, 579,  16078, 321, 1100, 9872, 303, 660, 17425};

struct Shape {
    const char* name;
    std::uint64_t ctx;
    std::uint64_t kv;
    int conc;
};

static const Shape kShapes[] = {
    {"C1 ", 73728, 73728, 1},
    {"C4 ", 73728, 73728, 4},
    {"T2 ", 512,   1024,  2},
    {"C4s", 512,   1024,  4},
    {"C8s", 512,   1024,  8},
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: oracle_shapes <artifact>\n";
        return 2;
    }
    int rc = 0;
    for (const auto& s : kShapes) {
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = s.ctx;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(s.kv);
        options.prefill_chunk = 256;
        options.speculative.backend = ninfer::SpeculativeBackend::None;
        options.speculative.draft_tokens = 0;
        options.enable_vision = false;
        options.use_cuda_graph = true;
        options.max_concurrency = s.conc;
        options.max_pending_requests = s.conc;
        options.context_cache.max_private_continuations = s.conc;
        options.context_cache.max_shared_prefixes = 1;

        try {
            ninfer::Engine engine(std::move(options));
            ninfer::RequestOptions req;
            req.execution.requested_output_tokens = 12;
            req.execution.sampling.temperature = 0.0F;
            req.execution.allow_prefix_reuse = false;
            req.stop.include_model_defaults = false;
            const auto result = engine.generate(engine.prepare_tokens(kPrompt), req);
            const auto& out = result.generated_token_ids;
            const bool pass = out == kGolden;
            std::cout << s.name << " ctx=" << s.ctx << " kv=" << s.kv
                      << " conc=" << s.conc << " " << (pass ? "PASS" : "DIVERGE");
            std::cout << " got:";
            for (auto t : out) std::cout << ' ' << t;
            if (!pass) {
                std::cout << " want:";
                for (auto t : kGolden) std::cout << ' ' << t;
                rc = 1;
            }
            std::cout << '\n' << std::flush;
        } catch (const std::exception& e) {
            std::cout << s.name << " ctx=" << s.ctx << " kv=" << s.kv << " conc=" << s.conc
                      << " ERROR " << e.what() << '\n' << std::flush;
            rc = 1;
        }
    }
    return rc;
}
