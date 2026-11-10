// A plan that contains ONLY a sliced request must still bind.
//
// The defect this pins: `plan_materialization` registers object storage only for whole-object
// requests (`placement.cpp:117`), and `bind_view` resolves a parent by object handle. A sliced
// request therefore has no object storage, and binding by handle either throws or -- worse -- binds
// the wrong region when the slice happens to be large enough. The existing slice test hides this by
// requesting the whole object in the same plan, so the object gets registered by the other request.
// This test has no whole-object request, and it checks the bound values, not just that binding
// succeeds.
//
// Host-only: the request is a host placement.

#include "artifact/encoded_slice.h"
#include "artifact/fixture.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/common/bindings.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
namespace af = ninfer::test::artifact_fixture;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    try {
        constexpr std::uint64_t rows = 384;
        constexpr std::uint64_t columns = 256;
        constexpr std::uint64_t half = columns / 2;

        // One BF16 object whose every byte is a function of its offset, so a wrong region cannot
        // match.
        af::Fixture f;
        const auto geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous, Shape{rows, columns});
        f.payload.resize(geometry.bytes);
        for (std::size_t i = 0; i < f.payload.size(); ++i) {
            f.payload[i] = std::byte((i * 131 + 7) % 251);
        }
        f.root["components"] = {{"text", {{"config", Json::object()}}}};
        f.root["objects"]    = Json::array({{{"id", "source"},
                                             {"kind", "tensor"},
                                             {"shape", Json::array({rows, columns})},
                                             {"format", "bf16"},
                                             {"layout", "contiguous_le_v1"},
                                             {"offset", 0},
                                             {"bytes", geometry.bytes}}});
        f.root["bindings"]   = {{"source", {{"object", "source"}}}};
        f.root["uses"]       = Json::array();
        f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", geometry.bytes}}});
        f.write();

        Reader reader(f.entry);
        const auto object = reader.find("source");

        // The rank's column half: one slice per row. No whole-object request anywhere in the plan.
        std::vector<MatrixSlice> slices;
        for (std::uint64_t row = 0; row < rows; ++row) {
            slices.push_back({object, row, half, 1, half, row, 0});
        }
        const Shape rank_shape{rows, half};
        std::vector<TensorRequest> requests{
            {slice_tensor(reader, rank_shape, slices), Device::cpu()}};
        auto plan = plan_materialization(reader, requests);
        if (plan.tensors.size() != 1) { return fail("the plan did not hold one sliced request"); }

        auto backing = materialize(reader, std::move(plan));

        // Bind by the plan's request, which is what a sliced plan must do.
        ParameterReference reference;
        reference.name  = "text/layers/0/attention/output";
        reference.shape = {rows, half};
        std::vector<models::PendingWeight> pending;
        pending.push_back({reference, {}, {}});
        const std::vector<Device> devices{Device::cpu()};
        const std::vector<models::WeightStorage> storage{
            {StorageId{0}, 0, rows * half}};
        const auto bound = models::resolve_weights(std::move(pending), backing, devices, storage);

        if (bound.size() != 1) { return fail("binding produced no weight"); }
        const auto& view = bound.front().view;
        if (view.parts.size() != 1) { return fail("the bound view is not one region"); }
        if (view.parts.front().parent->geometry.elements != rows * half) {
            return fail("the bound parent is not the rank's shape");
        }

        // The value check: every element must be the source's element at [row, half + column].
        const auto* got = static_cast<const std::byte*>(view.parts.front().parent->data);
        const auto* want = f.payload.data();
        const std::size_t element = geometry.row_bytes / columns;
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t column = 0; column < half; ++column) {
                const std::size_t destination = (row * half + column) * element;
                const std::size_t source     = (row * columns + half + column) * element;
                if (std::memcmp(got + destination, want + source, element) != 0) {
                    return fail("value differs at row " + std::to_string(row) + " column " +
                                std::to_string(column));
                }
            }
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
