// Every width>1 option combination either refuses with a named reason or is recorded as validated.
// The table lives in normalize_engine_options; this test is the register's teeth: each unvalidated
// combination must refuse naming its option, and the same option at width 1 must be accepted so the
// gate cannot creep into the single-device lane.
#include "runtime/engine/startup.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void fail(const std::string& message) {
    std::cerr << message << '\n';
    ++failures;
}

// Width 2: the combination must refuse, and the message must name the option it is refusing.
void expect_refusal(ninfer::EngineOptions options, const char* option) {
    try {
        (void)ninfer::runtime::normalize_engine_options(std::move(options));
        fail(std::string(option) + " was accepted at width 2 without a validated round");
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find(option) == std::string::npos) {
            fail(std::string(option) + " refused without naming itself: " + message);
        }
        if (message.find("no validated two-rank round") == std::string::npos) {
            fail(std::string(option) + " refused without saying why: " + message);
        }
    }
}

// Width 1: the same combination must be accepted untouched.
void expect_accepted(ninfer::EngineOptions options, const char* option) {
    try {
        (void)ninfer::runtime::normalize_engine_options(std::move(options));
    } catch (const std::invalid_argument& error) {
        fail(std::string(option) + " was refused at width 1: " + error.what());
    }
}

// The lane's own configuration: CUDA graph capture is refused at width 2 before the option rows are
// reached, so a width-2 engine must be configured eager. Leaving the default on would test the graph
// refusal seven times instead of the table.
ninfer::EngineOptions two_devices() {
    ninfer::EngineOptions options;
    options.devices               = {0, 1};
    options.use_cuda_graph        = false;
    // The context cache is on by default and is itself an unvalidated width-2 combination, so it
    // would refuse first for every other row. Each case turns off the rows it is not testing.
    options.context_cache.enabled = false;
    return options;
}

ninfer::EngineOptions one_device() {
    ninfer::EngineOptions options;
    options.devices = {0};
    return options;
}

} // namespace

int main() {
    {
        auto options = two_devices();
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        expect_refusal(options, "--spec mtp");
    }
    {
        auto options = two_devices();
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        expect_refusal(options, "--spec dflash");
    }
    {
        auto options = two_devices();
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        expect_refusal(options, "--spec dflash2");
    }
    {
        auto options   = two_devices();
        options.purpose = ninfer::EnginePurpose::CausalScoring;
        expect_refusal(options, "--causal-scoring");
    }
    {
        auto options                  = two_devices();
        options.context_cache.enabled = true;
        expect_refusal(options, "prefix reuse / context cache");
    }
    {
        auto options      = two_devices();
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        expect_refusal(options, "--kv-dtype (any quantized KV)");
    }
    {
        auto options        = two_devices();
        options.enable_vision = true;
        expect_refusal(options, "--vision");
    }

    // Width 1 keeps every one of those options: the gate lives inside the width>1 branch.
    {
        auto options = one_device();
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        expect_accepted(options, "--spec mtp");
    }
    {
        auto options = one_device();
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        expect_accepted(options, "--spec dflash2");
    }
    {
        auto options            = one_device();
        options.context_cache.enabled = true;
        expect_accepted(options, "prefix reuse / context cache");
    }
    {
        auto options      = one_device();
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        expect_accepted(options, "--kv-dtype (any quantized KV)");
    }
    {
        auto options        = one_device();
        options.enable_vision = true;
        expect_accepted(options, "--vision");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
