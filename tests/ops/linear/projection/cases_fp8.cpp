#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/linear/projection/runner.h"
#include <array>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

std::vector<Invocation> a16_capacity_calls() {
    std::vector<Invocation> calls;
    for (int t = 1; t <= 33; ++t) calls.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) calls.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 25, 33})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    for (int t : {64, 97, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    return calls;
}

// THE SHAPES A TENSOR-PARALLEL WIDTH-2 RANK PRESENTS, and the reason they are listed here rather
// than derived: every one of them is its own schedule set in the FP8 shape table, chosen by hand for
// its own (n, k). At width 2 the 27B's model-total FP8 objects become one rank's half of each — the
// row-sharded fused views (7168, 8192, 124160), the per-role splits the execution profile makes of
// them (3072, 512, 5120, 8704) and the column-sharded outputs, where a Columns shard halves K
// instead of the rows (5120x3072, 5120x8704). A schedule derived by scaling a parent's token bands
// is exactly the kind of change that compiles, links, resolves, and then faults on the device, so
// each shape runs here through the token bands it can actually be called with.
struct WidthTwoShape {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kWidthTwoFp8Shapes{
    WidthTwoShape{3072, 5120, 871U},    // attention query/gate role, gdn z role
    WidthTwoShape{512, 5120, 873U},     // attention key/value role
    WidthTwoShape{5120, 5120, 875U},    // gdn query+key+value role
    WidthTwoShape{8704, 5120, 877U},    // mlp gate/up role
    WidthTwoShape{5120, 3072, 879U},    // attention/output and gdn/output (Columns shard halves K)
    WidthTwoShape{5120, 8704, 881U},    // mlp/down (Columns shard halves K)
    WidthTwoShape{124160, 5120, 883U},  // text/output_head (Rows shard)
    WidthTwoShape{7168, 5120, 885U},    // attention qkv+gate fused half
    WidthTwoShape{8192, 5120, 887U},    // gdn qkvz fused half
};

// The shapes the WIDTH-1 reference arm presents and the width-2 lane does not. They surfaced one
// run at a time as the reference arm reached them, which is the same gap in a different direction —
// a shape the table never had, so no case could miss it. They are listed apart from the width-2 set
// because they belong to a different presentation of the same weights, and a reader looking for
// "what does width 2 present" should not have to filter them out.
//
// Every one of them was derived from the artifact's own binding parts rather than from the recipe:
// the GDN input projection is one 16384-row object whose parts are contiguous in
// query|key|value|z order, so width 1 consumes it as a 10240-row qkv parent plus a 6144-row z
// parent; the attention input projection is one 14336-row object whose four parts are also
// contiguous, but the packed form still hands the model one weight per role, so width 1 consumes it
// as 6144 / 1024 / 6144 / 1024. N6144K5120 serves both the GDN z parent and the attention query and
// gate roles, because the FP8 table is keyed by (n, k) and not by role. The MLP roles are the one
// entry whose presentation the binding parts did NOT predict: the two parts are adjacent, but a
// row-scaled payload's scale plane sits after the whole code plane, so neither role is a contiguous
// parent region and the engine takes the two-role form (`ops::projection_region_count` counts byte
// regions, not element ranges). The NVFP4 table has served the same 17408-row shape all along.
constexpr std::array kWidthOneFp8Shapes{
    WidthTwoShape{10240, 5120, 891U},  // GDN qkv parent: query 2048 + key 2048 + value 6144
    WidthTwoShape{6144, 5120, 893U},   // GDN z parent; also the attention query and gate roles
    WidthTwoShape{1024, 5120, 895U},   // attention key and value roles: 4 KV heads of width 256
    WidthTwoShape{17408, 5120, 897U},  // dense MLP gate and up roles: half of the 34816-row parent
};

// The token extents that cross every band boundary of the width-2 shapes' A8 schedules: the A8
// threshold at 17, then the 64/128/192/256/384/512/768/1024 tile boundaries and one past each.
std::vector<Invocation> width_two_a8_calls() {
    std::vector<Invocation> calls;
    for (int t : {17, 33, 64, 65, 97, 128, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769,
                  1024, 1025, 2048})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    // The verify round is graph-captured, so the captured form runs too.
    for (int t : {128, 1025})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
    return calls;
}

void run_fp8_a16(Suite& suite) {
    const auto attn_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {14336, 5120, 811U, Comparison::Sampled, true, attn_invocations});
    const auto gdn_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {16384, 5120, 817U, Comparison::Sampled, true, gdn_invocations});
    const auto mlp_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight, {34816, 5120, 821U, Comparison::Sampled, true, mlp_invocations});
    std::vector<Invocation> vocabulary_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{96, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{97, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{160, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{161, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    for (int t = 1; t <= 41; ++t)
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    for (int t : {7, 25, 41, 65, 128})
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {248320, 5120, 823U, Comparison::Sampled, true, vocabulary_invocations});
    suite.add(linear_suite::make_case("fp8", 248320, 5120, 0, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "vocabulary-workspace"),
              [] {
                  int failures = 0;
                  const std::array vocabulary_policies{
                      ops::LinearPolicy::A16Only,
                      ops::LinearPolicy::AllowA8,
                      ops::LinearPolicy::AllowA4,
                  };
                  for (const ops::LinearPolicy policy : vocabulary_policies) {
                      try {
                          const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                              QType::FP8_E4M3FN_ROW_BF16, 248320, 5120, policy, 1, 2048);
                          if (capacity != 0) {
                              std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                              ++failures;
                          }
                      } catch (const std::exception& error) {
                          std::cerr << "FP8 vocabulary policy was rejected: " << error.what()
                                    << '\n';
                          ++failures;
                      }
                  }
                  return failures;
              });
    const auto residual6144_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 6144, 827U, Comparison::Sampled, true, residual6144_invocations});
    const auto residual17408_invocations = a16_capacity_calls();
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 17408, 829U, Comparison::Sampled, true, residual17408_invocations});

    // The width-2 shapes, through the A16 bands the decode and sampled-prefill calls land in.
    for (const auto& shape : kWidthTwoFp8Shapes) {
        const auto invocations = a16_capacity_calls();
        add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
                  make_fp8_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Sampled, true, invocations});
    }
    // The width-1 paired GDN parents, through the same A16 bands.
    for (const auto& shape : kWidthOneFp8Shapes) {
        const auto invocations = a16_capacity_calls();
        add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
                  make_fp8_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Sampled, true, invocations});
    }

    suite.add(linear_suite::make_case("fp8", 14336, 5120, 1, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "metadata-s831"),
              [] {
                  int failures = 0;
                  auto packed  = make_fp8_weight(14336, 5120, 831U);
                  DeviceBuffer device_weight(ninfer::test::cuda_environment().context.device(),
                                             packed.payload.size());
                  device_weight.copy_from_host(packed.payload.data(), device_weight.size());
                  const Weight weight = packed.device_weight(device_weight.span());
                  const std::vector<std::uint16_t> input_bits(5120, 0);
                  auto device_input = test::to_device(input_bits);
                  test::GuardedDeviceBuffer device_output(14336 * sizeof(std::uint16_t));
                  device_output.fill(0xff);
                  Tensor input(device_input.span(), DType::BF16, {5120, 1});
                  Tensor output(device_output.span(), DType::BF16, {14336, 1});
                  ninfer::DeviceBuffer workspace_storage(
                      ninfer::test::cuda_environment().context.device(), 256);
                  MemoryArena workspace(workspace_storage.span());
                  const auto invoke = [&](const Weight& candidate) {
                      ops::linear(input, candidate, output, ops::LinearPolicy::A16Only, workspace,
                                  ninfer::test::cuda_environment().default_execution);
                  };
                  try {
                      invoke(weight);
                      test::cuda_synchronize();
                  } catch (const std::exception& error) {
                      std::cerr << "valid FP8 metadata was rejected: " << error.what() << '\n';
                      ++failures;
                  }
                  const auto expect_invalid = [&](const char* label, Weight invalid) {
                      try {
                          invoke(invalid);
                          test::cuda_synchronize();
                          std::cerr << "invalid FP8 " << label << " was accepted\n";
                          ++failures;
                      } catch (const std::invalid_argument&) {}
                  };
                  Weight invalid = weight;
                  invalid.layout = QuantLayout::Contiguous;
                  expect_invalid("layout", invalid);
                  invalid             = weight;
                  invalid.scale_nb[1] = invalid.scale_nb[1] - 2;
                  expect_invalid("scale stride", invalid);
                  invalid               = weight;
                  invalid.payload_bytes = invalid.payload_bytes - 1;
                  expect_invalid("payload bound", invalid);
                  failures += device_output.verify_guards("FP8 metadata output");
                  failures += test::linear_common::verify_preserved(
                      device_input.data(), std::as_bytes(std::span(input_bits)),
                      "FP8 metadata input");
                  failures += test::linear_common::verify_preserved(
                      device_weight.data(), std::as_bytes(std::span(packed.payload)),
                      "FP8 metadata weight");
                  return failures;
              });
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{248320, 5120}, std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, n, k);
    }
    for (const auto& shape : kWidthTwoFp8Shapes) {
        add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, shape.n, shape.k);
    }
    for (const auto& shape : kWidthOneFp8Shapes) {
        add_workspace_cases(suite, QType::FP8_E4M3FN_ROW_BF16, shape.n, shape.k);
    }
}

} // namespace

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

void run_fp8_a8(Suite& suite) {
    const std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {14336, 5120, 827U, Comparison::Sampled, true, attn_decode_invocations});
    const std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {14336, 5120, 829U, Comparison::Sampled, true, attn_invocations});
    const std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {14336, 5120, 833U, Comparison::Sampled, true, attn_bulk_invocations});
    const std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {16384, 5120, 837U, Comparison::Sampled, true, gdn_decode_invocations});
    const std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {16384, 5120, 839U, Comparison::Sampled, true, gdn_invocations});
    const std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {16384, 5120, 841U, Comparison::Sampled, true, gdn_bulk_invocations});
    const std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {34816, 5120, 851U, Comparison::Sampled, true, mlp_decode_invocations});
    const std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {34816, 5120, 853U, Comparison::Sampled, true, mlp_invocations});

    const std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {34816, 5120, 863U, Comparison::Sampled, true, mlp_bulk_invocations});

    const std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 6144, 855U, Comparison::Sampled, true, residual6144_decode_invocations});
    const std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {5120, 6144, 857U, Comparison::Sampled, true, residual6144_invocations});
    const std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::TokenBlocks,
              make_fp8_weight,
              {5120, 17408, 867U, Comparison::Sampled, true, residual17408_decode_invocations});
    const std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed, make_fp8_weight,
              {5120, 17408, 859U, Comparison::Sampled, true, residual17408_invocations});

    // The width-2 shapes through the A8 bands, which the verify and prefill rounds land in. Same
    // shapes, different activation recipe and different token extents: the A16 and A8 halves of a
    // shape's schedule set are separate selections and each has to run.
    for (const auto& shape : kWidthTwoFp8Shapes) {
        const auto invocations = width_two_a8_calls();
        add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
                  make_fp8_weight,
                  {shape.n, shape.k, shape.seed + 100U, Comparison::Sampled, true, invocations});
    }
    // The width-1 paired GDN parents through the same A8 bands.
    for (const auto& shape : kWidthOneFp8Shapes) {
        const auto invocations = width_two_a8_calls();
        add_shape(suite, ninfer::QType::FP8_E4M3FN_ROW_BF16, ActivationRecipe::Hashed,
                  make_fp8_weight,
                  {shape.n, shape.k, shape.seed + 100U, Comparison::Sampled, true, invocations});
    }
}

} // namespace

void ninfer::test::linear::add_fp8_cases(Suite& suite) {
    run_fp8_a16(suite);
    run_fp8_a8(suite);
}
