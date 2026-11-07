// SP-4 adaptive tiers: the request-aware expert-usage policy.
//
// Tracks which experts each session's requests exercise, caches the per-session expert
// sets by the context-cache session key, and ranks experts by cumulative usage so the
// runtime can pin the hot set. The policy owns no device state: the engine feeds it the
// per-layer counter readbacks and consumes its ranked lists against the live residency.
#pragma once

#include <array>
#include <list>
#include <cmath>
#include <mutex>
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
    // usage ledger and the session's cached expert set. Layer 0 updates the request stats.
    // Returns true when this layer's counts diverge from the class's running centroid by
    // more than the threshold (cosine distance): the request is atypical for the class
    // and the caller should also store it under a fingerprint-derived key.
    bool observe(std::string_view session, int layer, std::span<const std::uint32_t> counts) {
        const std::lock_guard lock{mutex_};
        return observe_unlocked(session, layer, counts, 0.0);
    }

    bool observe(std::string_view session, int layer, std::span<const std::uint32_t> counts,
                 double split_threshold) {
        const std::lock_guard lock{mutex_};
        return observe_unlocked(session, layer, counts, split_threshold);
    }

private:
    // Both public overloads hold mutex_; the body is shared here. Never call with the
    // mutex free from a public path that already locked it.
    bool observe_unlocked(std::string_view session, int layer,
                          std::span<const std::uint32_t> counts, double split_threshold) {
        if (layer == 0) {
            stats_.requests += 1;
            if (cache_.find(std::string(session)) != cache_.end()) { stats_.cache_hits += 1; }
        }
        auto& entry = touch(session);
        // Observations counted per record, not per request: fingerprint and layer>0
        // entries never pass through layer 0, so entry.requests stays 0 there and the
        // centroid denominator must not read it (the old seen = requests - 1 wrapped).
        const auto seen = static_cast<double>(entry.observations);
        entry.observations += 1;
        if (layer == 0) { entry.requests += 1; }
        const auto base = static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_);
        const auto count = static_cast<std::size_t>(experts_) < counts.size()
                               ? static_cast<std::size_t>(experts_)
                               : counts.size();
        auto& mask = entry.mask[static_cast<std::size_t>(layer)];
        mask.resize(static_cast<std::size_t>(experts_), 0);
        auto& usage = entry.usage[static_cast<std::size_t>(layer)];
        usage.resize(static_cast<std::size_t>(experts_), 0);
        // Divergence against the running centroid (the mean of prior request vectors):
        // the cosine distance between this request's count vector and the centroid.
        auto& centroid = entry.centroid[static_cast<std::size_t>(layer)];
        centroid.resize(static_cast<std::size_t>(experts_), 0.0);
        double divergent = false;
        if (split_threshold > 0.0 && entry.requests > 1) {
            double dot = 0.0, norm_c = 0.0, norm_r = 0.0;
            for (std::size_t e = 0; e < count; ++e) {
                dot += centroid[e] * static_cast<double>(counts[e]);
                norm_c += centroid[e] * centroid[e];
                norm_r += static_cast<double>(counts[e]) * counts[e];
            }
            if (norm_c > 0.0 && norm_r > 0.0) {
                const double cosine = dot / (std::sqrt(norm_c) * std::sqrt(norm_r));
                divergent = (1.0 - cosine) > split_threshold;
            }
        }
        for (std::size_t e = 0; e < count; ++e) {
            usage_[base + e] += counts[e];
            usage[e] += counts[e];
            centroid[e] = (centroid[e] * seen + static_cast<double>(counts[e])) / (seen + 1.0);
            if (counts[e] != 0) { mask[e] = 1; }
        }
        return divergent;
    }

public:

    // The `count` experts of one layer most-exercised by this class, most first. Feeds
    // the burst prefetch: the limited budget goes to the load-bearing experts first.
    [[nodiscard]] std::vector<int> class_top(std::string_view session, int layer,
                                             int count) const {
        const std::lock_guard lock{mutex_};
        const auto found = cache_.find(std::string(session));
        std::vector<int> out;
        if (found == cache_.end() || layer >= layers_ || count <= 0) { return out; }
        const auto& usage = found->second.usage[static_cast<std::size_t>(layer)];
        std::vector<int> order(usage.size());
        for (std::size_t e = 0; e < usage.size(); ++e) { order[e] = static_cast<int>(e); }
        std::partial_sort(order.begin(),
                          order.begin() + std::min<std::size_t>(static_cast<std::size_t>(count),
                                                                order.size()),
                          order.end(), [&usage](int a, int b) {
                              return usage[static_cast<std::size_t>(a)] >
                                     usage[static_cast<std::size_t>(b)];
                          });
        for (const int e : order) {
            if (static_cast<int>(out.size()) >= count) { break; }
            if (usage[static_cast<std::size_t>(e)] > 0) { out.push_back(e); }
        }
        return out;
    }

    // True when the session's expert sets are cached (the repeat request).
    [[nodiscard]] bool cached(std::string_view session) const {
        const std::lock_guard lock{mutex_};
        return cache_.find(std::string(session)) != cache_.end();
    }

    // The session's cached expert bitmask for one layer; empty when uncached.
    [[nodiscard]] const std::vector<std::uint8_t>* cached_mask(std::string_view session,
                                                               int layer) const {
        const std::lock_guard lock{mutex_};
        const auto found = cache_.find(std::string(session));
        if (found == cache_.end() || layer >= layers_) { return nullptr; }
        return &found->second.mask[static_cast<std::size_t>(layer)];
    }

    // The `count` most-used experts of one layer, most-used first.
    [[nodiscard]] std::vector<int> top_experts(int layer, int count) const {
        const std::lock_guard lock{mutex_};
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

    // The router's decay trick: every `interval` observed requests, halve the usage
    // ledger. Recently-quiet experts fade toward zero and fall out of the top ranking, so
    // the global pin pass re-balances the hot tier toward the current traffic mix without
    // any explicit eviction logic. Call after the per-layer observe loop.
    void age_if_due(std::size_t current_requests, int interval) {
        const std::lock_guard lock{mutex_};
        if (interval <= 0 || current_requests == 0) { return; }
        if (current_requests / static_cast<std::size_t>(interval) <= aged_periods_) { return; }
        aged_periods_ = current_requests / static_cast<std::size_t>(interval);
        for (auto& value : usage_) {
            value /= 2;
        }
    }

    // The number of requests attributed to this class so far (0 when unknown). The
    // burst-budget decision uses it to front-load a new class's ratchet.
    [[nodiscard]] std::size_t class_requests(std::string_view session) const {
        const std::lock_guard lock{mutex_};
        const auto found = cache_.find(std::string(session));
        return found == cache_.end() ? 0 : found->second.requests;
    }

    // The union of every cached session's expert bitmask for one layer. A global demotion
    // pass must not evict experts any known session needs.
    [[nodiscard]] std::vector<std::uint8_t> all_sessions_mask(int layer) const {
        const std::lock_guard lock{mutex_};
        std::vector<std::uint8_t> out;
        if (layer >= layers_) { return out; }
        out.assign(static_cast<std::size_t>(experts_), 0);
        for (const auto& [key, entry] : cache_) {
            const auto& mask = entry.mask[static_cast<std::size_t>(layer)];
            const auto count = std::min(mask.size(), out.size());
            for (std::size_t e = 0; e < count; ++e) {
                out[e] = static_cast<std::uint8_t>(out[e] | mask[e]);
            }
        }
        return out;
    }

    // The `count` least-used experts of one layer among `eligible` (the caller's currently
    // hot set), excluding the protected set (the pinned or the always-hot experts the
    // caller must not demote).
    [[nodiscard]] std::vector<int> bottom_experts(int layer, int count,
                                                  const std::vector<std::uint8_t>& eligible,
                                                  const std::vector<std::uint8_t>& protect) const {
        const std::lock_guard lock{mutex_};
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

        [[nodiscard]] Stats stats() const {
        const std::lock_guard lock{mutex_};
        return stats_;
    }

private:
    struct SessionEntry {
        // Per-layer expert bitmasks: 1 = the session exercised this expert at least once.
        std::vector<std::vector<std::uint8_t>> mask;
        // Per-layer cumulative usage counts, indexed like mask. Ranks the prefetch's
        // promote candidates: with a burst budget the most-exercised experts go first.
        std::vector<std::vector<std::uint64_t>> usage;
        // Per-layer running centroid of the per-request count vectors, for the
        // divergence-based class split.
        std::vector<std::vector<double>> centroid;
        // How many requests have been attributed to this class (layer-0 observes).
        std::size_t requests = 0;
        // How many records have landed in this entry's counters (every observe call).
        // Layer>0 and fingerprint entries never pass layer 0, so this is the only
        // sound centroid denominator for them.
        std::size_t observations = 0;
    };

    SessionEntry& touch(std::string_view session) {
        const std::string key(session);
        if (auto found = cache_.find(key); found != cache_.end()) {
            for (auto it = touch_order_.begin(); it != touch_order_.end(); ++it) {
                if (*it == key) {
                    touch_order_.splice(touch_order_.end(), touch_order_, it);
                    break;
                }
            }
            return found->second;
        }
        while (cache_.size() >= cache_capacity_ && !touch_order_.empty()) {
            const auto& oldest = touch_order_.front();
            cache_.erase(oldest);
            touch_order_.pop_front();
            stats_.evictions += 1;
        }
        touch_order_.emplace_back(key);
        SessionEntry entry;
        entry.mask.resize(static_cast<std::size_t>(layers_));
        entry.usage.resize(static_cast<std::size_t>(layers_));
        entry.centroid.resize(static_cast<std::size_t>(layers_));
        return cache_.emplace(key, std::move(entry)).first->second;
    }

    // The policy is shared between two threads: the submit thread runs the prefetch
    // (tier_prefetch) while the worker thread runs the attribution
    // (tier_observe_request). Every public method takes this mutex - an unsynchronized
    // observe during a prefetch torn an unordered_map rehash and hung the submit thread
    // in _Hashtable::find (the 5090 handover's M1).
    mutable std::mutex mutex_;

    int layers_ = 0;
    int experts_ = 0;
    std::size_t cache_capacity_ = 0;
    std::vector<std::uint64_t> usage_;
    std::unordered_map<std::string, SessionEntry> cache_;
    std::list<std::string> touch_order_;
    std::size_t aged_periods_ = 0;
    Stats stats_{};
};

} // namespace ninfer::flashnext_runtime
