// SP-4 adaptive tiers: the request-aware expert-usage policy.
//
// Tracks which experts each session's requests exercise, caches the per-session expert
// sets by the context-cache session key, and ranks experts by cumulative usage so the
// runtime can pin the hot set. The policy owns no device state: the engine feeds it the
// per-layer counter readbacks and consumes its ranked lists against the live residency.
#pragma once

#include <array>
#include <list>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::flashnext_runtime {

class TierPolicy {
public:
    TierPolicy(int layers, int experts, std::size_t cache_capacity)
        : layers_(layers), experts_(experts), cache_capacity_(cache_capacity),
          usage_(static_cast<std::size_t>(layers) * static_cast<std::size_t>(experts), 0) {}

    // Attributes one layer's counter readback (the deltas since the last reset) to the
    // session. Call once per layer per completed request; the counts accumulate into the
    // usage ledger and the session's cached expert set.
    void observe(std::string_view session, int layer, std::span<const std::uint32_t> counts) {
        auto& entry = touch(session);
        const auto base = static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_);
        const auto count = static_cast<std::size_t>(experts_) < counts.size()
                               ? static_cast<std::size_t>(experts_)
                               : counts.size();
        auto& mask = entry.mask[static_cast<std::size_t>(layer)];
        mask.resize(static_cast<std::size_t>(experts_), 0);
        for (std::size_t e = 0; e < count; ++e) {
            usage_[base + e] += counts[e];
            if (counts[e] != 0) { mask[e] = 1; }
        }
    }

    // True when the session's expert sets are cached (the repeat request).
    [[nodiscard]] bool cached(std::string_view session) const {
        return cache_.find(std::string(session)) != cache_.end();
    }

    // The session's cached expert bitmask for one layer; empty when uncached.
    [[nodiscard]] const std::vector<std::uint8_t>* cached_mask(std::string_view session,
                                                               int layer) const {
        const auto found = cache_.find(std::string(session));
        if (found == cache_.end() || layer >= layers_) { return nullptr; }
        return &found->second.mask[static_cast<std::size_t>(layer)];
    }

    // The `count` most-used experts of one layer, most-used first.
    [[nodiscard]] std::vector<int> top_experts(int layer, int count) const {
        const auto base = static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_);
        std::vector<int> order(static_cast<std::size_t>(experts_));
        for (int e = 0; e < experts_; ++e) { order[static_cast<std::size_t>(e)] = e; }
        const auto& usage = usage_;
        std::partial_sort(order.begin(), order.begin() + count, order.end(),
                          [base, usage](int a, int b) {
                              return usage[base + static_cast<std::size_t>(a)] >
                                     usage[base + static_cast<std::size_t>(b)];
                          });
        order.resize(static_cast<std::size_t>(count));
        return order;
    }

    // The `count` least-used experts of one layer among `eligible` (the caller's currently
    // hot set), excluding the protected set (the pinned or the always-hot experts the
    // caller must not demote).
    [[nodiscard]] std::vector<int> bottom_experts(int layer, int count,
                                                  const std::vector<std::uint8_t>& eligible,
                                                  const std::vector<std::uint8_t>& protect) const {
        const auto base = static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_);
        std::vector<int> order(static_cast<std::size_t>(experts_));
        for (int e = 0; e < experts_; ++e) { order[static_cast<std::size_t>(e)] = e; }
        const auto& usage = usage_;
        std::sort(order.begin(), order.end(),
                  [base, usage](int a, int b) {
                      return usage[base + static_cast<std::size_t>(a)] <
                             usage[base + static_cast<std::size_t>(b)];
                  });
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(count));
        for (const int e : order) {
            const auto idx = static_cast<std::size_t>(e);
            const bool eligible_e = eligible.empty() || eligible[idx] != 0;
            const bool protected_e = !protect.empty() && protect[idx] != 0;
            if (eligible_e && !protected_e) {
                out.push_back(e);
                if (static_cast<int>(out.size()) >= count) { break; }
            }
        }
        return out;
    }

    struct Stats {
        std::size_t requests = 0;
        std::size_t cache_hits = 0;
        std::size_t evictions = 0;
    };

    [[nodiscard]] Stats stats() const noexcept { return stats_; }

private:
    struct SessionEntry {
        // Per-layer expert bitmasks: 1 = the session exercised this expert at least once.
        std::vector<std::vector<std::uint8_t>> mask;
    };

    SessionEntry& touch(std::string_view session) {
        stats_.requests += 1;
        const std::string key(session);
        if (auto found = cache_.find(key); found != cache_.end()) {
            stats_.cache_hits += 1;
            for (auto it = touch_order_.begin(); it != touch_order_.end(); ++it) {
                if (*it == key) {
                    touch_order_.splice(touch_order_.end(), touch_order_, it);
                    break;
                }
            }
            return found->second;
        }
        stats_.evictions += 1;
        while (cache_.size() >= cache_capacity_ && !touch_order_.empty()) {
            const auto& oldest = touch_order_.front();
            cache_.erase(oldest);
            touch_order_.pop_front();
        }
        touch_order_.emplace_back(key);
        SessionEntry entry;
        entry.mask.resize(static_cast<std::size_t>(layers_));
        return cache_.emplace(key, std::move(entry)).first->second;
    }

    int layers_ = 0;
    int experts_ = 0;
    std::size_t cache_capacity_ = 0;
    std::vector<std::uint64_t> usage_;
    std::unordered_map<std::string, SessionEntry> cache_;
    std::list<std::string> touch_order_;
    Stats stats_{};
};

} // namespace ninfer::flashnext_runtime
