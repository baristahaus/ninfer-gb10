#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace q38 = ninfer::models::qwen3_8_flash_next;

int main() {
    try {
        const std::array<std::int32_t, 6> tokens = {1, 2, 3, q38::kPleEosToken, 4, 5};
        std::array<q38::PleIds, tokens.size()> ids{};
        q38::compute_ple_ids(tokens, ids);
        const std::array<q38::PleIds, tokens.size()> expected = {{
            {16121432, 28938500, 59087997, 73487090, 81148277, 104500129, 120276032,
             149373875, 176283436, 184305849, 216528839, 231080079, 257961536, 266068568,
             289043455, 305959965},
            {15220011, 32170723, 40646129, 76511807, 98682440, 106072663, 127158041,
             141938523, 170367010, 180520041, 210849650, 234005644, 252857364, 272068622,
             291885712, 311569748},
            {14605717, 24410875, 49313567, 72177428, 86060820, 104022010, 130963819,
             146886202, 161523077, 195939126, 219424565, 220811782, 248020141, 275228530,
             284297999, 309645223},
            {1040936, 20493796, 54629949, 79359381, 85142054, 114677089, 129861487,
             151753631, 160770146, 187379937, 207108329, 229509886, 243927029, 279402641,
             291463027, 313528485},
            {16786187, 37399507, 51447157, 75303773, 99642929, 108554057, 122668943,
             142885423, 178075680, 189995935, 213432942, 234309713, 243139806, 273195768,
             283486804, 316984683},
            {8807125, 30450454, 51272170, 64422588, 81408710, 113737508, 122230591,
             146888097, 162512814, 189149098, 215375741, 226808975, 243902281, 261355125,
             280586077, 317339594},
        }};
        if (ids != expected) { throw std::runtime_error("PLE n-gram IDs differ from oracle"); }
        for (const ninfer::DType dtype : {ninfer::DType::FP8_E4M3FN, ninfer::DType::BF16}) {
            const std::size_t row_bytes = 160U * ninfer::dtype_size(dtype);
            const std::uint64_t table_bytes = q38::kPleRows * row_bytes;
            auto first = std::make_shared<std::vector<std::byte>>(2 * row_bytes);
            auto last = std::make_shared<std::vector<std::byte>>(2 * row_bytes);
            for (std::size_t offset = 0; offset < row_bytes; ++offset) {
                (*first)[offset] = std::byte(offset & 255U);
                (*last)[row_bytes + offset] = std::byte((offset * 3U + 1U) & 255U);
            }
            ninfer::artifact::MappedRange table;
            table.segments = {
                {std::shared_ptr<const std::byte>(first, first->data()), 0, 2 * row_bytes},
                {std::shared_ptr<const std::byte>(last, last->data()),
                 table_bytes - 2 * row_bytes, 2 * row_bytes},
            };
            std::array<q38::PleIds, 2> selected{};
            selected[1].fill(static_cast<std::uint32_t>(q38::kPleRows - 1));
            std::vector<std::byte> result(2 * q38::kPleEmbeddingDim *
                                          ninfer::dtype_size(dtype));
            q38::gather_ple(table, dtype, selected, result);
            for (std::size_t head = 0; head < q38::kPleHeads; ++head) {
                for (std::size_t offset = 0; offset < row_bytes; ++offset) {
                    if (result[head * row_bytes + offset] != (*first)[offset] ||
                        result[(q38::kPleHeads + head) * row_bytes + offset] !=
                            (*last)[row_bytes + offset]) {
                        throw std::runtime_error("PLE gathered row differs from source bytes");
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
