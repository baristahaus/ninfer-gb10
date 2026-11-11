#include "models/common/qwen/frontend/processor_impl.h"
#include "models/common/qwen/frontend/processor_family.h"

#include <sstream>

namespace ninfer::models::qwen::frontend {

std::string PreprocessStats::summary() const {
    std::ostringstream out;
    out << "media=" << media_items << " media_bytes=" << media_bytes << " patches=" << raw_patches
        << " vision_tokens=" << vision_tokens << " attention_pairs=" << attention_pairs
        << " prompt_tokens=" << prompt_tokens << " patch_bytes=" << patch_bytes;
    return out.str();
}

std::span<const std::int32_t> ProcessedInput::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != input_ids.size() * 3) {
        throw std::out_of_range("invalid processed position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * input_ids.size(), input_ids.size());
}

template class Processor<CommonProcessorFamily>;

} // namespace ninfer::models::qwen::frontend
