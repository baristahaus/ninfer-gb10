#include "ops/linear/add/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
LinearAddLaunch select_q5_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (tokens > 0) {
        if (rows == 5120 && k == 6144) return select_q5_linear_add_n5120_k6144(tokens);
        if (rows == 5120 && k == 17408) return select_q5_linear_add_n5120_k17408(tokens);
    }
    throw std::invalid_argument("q5 linear_add: unsupported shape or token extent");
}

LinearAddLaunch select_q8_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (tokens > 0) {
        if (rows == 2048 && k == 4096) return select_q8_linear_add_n2048_k4096(tokens);
        if (rows == 2048 && k == 6144) return select_q8_linear_add_n2048_k6144(tokens);
        if (rows == 5120 && k == 6144) return select_q8_linear_add_n5120_k6144(tokens);
        if (rows == 5120 && k == 17408) return select_q8_linear_add_n5120_k17408(tokens);
        // The width-2 Columns shards. The fused add follows the projection: when a Columns shard
        // halves K, the residual add that fuses into it presents the halved K too, so the table needs
        // the rank-sized geometry as well as the width-1 one. mlp/down presents 5120x8704 and
        // attention out 5120x3072; both were missing, which is what a width-2 MTP round measured as
        // 'q8 linear_add: unsupported shape or token extent N5120K8704 T1'.
        if (rows == 5120 && k == 8704) return select_q8_linear_add_n5120_k8704(tokens);
        if (rows == 5120 && k == 3072) return select_q8_linear_add_n5120_k3072(tokens);
    }
    throw std::invalid_argument("q8 linear_add: unsupported shape or token extent");
}

} // namespace ninfer::ops::detail
