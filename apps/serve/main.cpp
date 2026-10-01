#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

// Signals only count; the drain thread acts on them. A lock-free atomic is async-signal-safe,
// and a third signal leaves at once for an operator whose drain is stuck.
std::atomic<int> g_signals{0};
static_assert(std::atomic<int>::is_always_lock_free);

void handle_signal(int) {
    if (g_signals.fetch_add(1) + 1 >= 3) { std::_Exit(130); }
}

// SIGINT/SIGTERM: stop admitting, let admitted requests finish for the configured timeout (or
// until a second signal), cancel the rest, then stop the HTTP server so listen() returns.
void run_drain(ninfer::serve::HttpServer& server, ninfer::serve::GenerationService& service,
               const ninfer::serve::OperationalLog& log, std::uint32_t timeout_seconds,
               const std::stop_token& stop) {
    using Clock          = std::chrono::steady_clock;
    constexpr auto kPoll = std::chrono::milliseconds(100);
    while (g_signals.load() == 0) {
        if (stop.stop_requested()) { return; }
        std::this_thread::sleep_for(kPoll);
    }
    const Clock::time_point started = Clock::now();
    log.drain_started(server.in_flight(), timeout_seconds);
    server.begin_drain();
    const Clock::time_point deadline = started + std::chrono::seconds(timeout_seconds);
    while (server.in_flight() != 0 && Clock::now() < deadline && g_signals.load() < 2) {
        (void)server.wait_for_idle(std::min(deadline, Clock::now() + kPoll));
    }
    if (server.in_flight() != 0) {
        log.drain_cancelling(server.in_flight(), g_signals.load() >= 2);
        service.cancel_running();
        // Cancellation takes effect at the next Engine round boundary.
        (void)server.wait_for_idle(Clock::now() + std::chrono::seconds(10));
    }
    log.drain_finished(std::chrono::duration<double>(Clock::now() - started).count(),
                       server.in_flight());
    server.stop();
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        std::optional<ninfer::serve::GenerationService> service_storage;
        service_storage.emplace(options, startup_log.observer());
        ninfer::serve::GenerationService& service = *service_storage;
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::jthread drain([&](std::stop_token stop) {
            run_drain(server, service, operational_log, options.shutdown_timeout_seconds, stop);
        });

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        drain.request_stop();
        drain.join();
        // Release the Engine's device memory before exit, so a restarted server finds it free
        // (on an integrated device the kernel otherwise returns it after the process ends).
        const auto release_started = std::chrono::steady_clock::now();
        service_storage.reset();
        operational_log.engine_released(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - release_started)
                .count());
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
