#include "serve/cors.h"

#include <httplib.h>

#include <iostream>
#include <string>
#include <thread>

namespace {

using ninfer::serve::answer_cors_preflight;
using ninfer::serve::apply_cors_defaults;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool allows(const httplib::Response& response, const char* header) {
    const std::string allowed = response.get_header_value("Access-Control-Allow-Headers");
    return allowed.find(header) != std::string::npos;
}

} // namespace

int main() {
    httplib::Server server;
    apply_cors_defaults(server);
    server.Options(R"(.*)", [](const httplib::Request& request, httplib::Response& response) {
        answer_cors_preflight(request, response);
    });
    server.Post("/v1/chat/completions", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{}", "application/json");
    });
    server.Post("/v1/stream", [](const httplib::Request&, httplib::Response& response) {
        response.set_chunked_content_provider("text/event-stream",
                                              [](size_t, httplib::DataSink& sink) {
                                                  sink.write("data: x\n\n", 9);
                                                  sink.done();
                                                  return true;
                                              });
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread worker([&] { server.listen_after_bind(); });
    httplib::Client client("127.0.0.1", port);
    int failures = 0;

    // A browser's preflight for an OpenAI or Anthropic SDK request, with Chromium's Private
    // Network Access header.
    const httplib::Headers preflight_headers = {
        {"Origin", "https://app.example"},
        {"Access-Control-Request-Method", "POST"},
        {"Access-Control-Request-Headers", "authorization, content-type, x-stainless-os, "
                                           "anthropic-dangerous-direct-browser-access"},
        {"Access-Control-Request-Private-Network", "true"},
    };
    if (auto response = client.Options("/v1/chat/completions", preflight_headers)) {
        failures += check(response->status == 204, "preflight is not 204");
        failures += check(response->get_header_value("Access-Control-Allow-Origin") == "*",
                          "preflight lacks Access-Control-Allow-Origin");
        failures += check(response->get_header_value("Access-Control-Max-Age") == "600",
                          "preflight lacks Access-Control-Max-Age");
        failures +=
            check(response->get_header_value("Access-Control-Allow-Private-Network") == "true",
                  "preflight did not acknowledge Private Network Access");
        failures += check(allows(*response, "x-stainless-os") &&
                              allows(*response, "anthropic-dangerous-direct-browser-access") &&
                              allows(*response, "Authorization"),
                          "preflight does not allow the SDK headers a browser sends");
    } else {
        failures += check(false, "preflight request failed");
    }

    // The private-network echo is requested only by Chromium; without it, no grant is sent.
    if (auto response = client.Options("/v1/chat/completions", {{"Origin", "https://app"}})) {
        failures +=
            check(response->get_header_value("Access-Control-Allow-Private-Network").empty(),
                  "private-network grant sent without a request");
    }

    // Every application response carries the grant, including a streamed one.
    if (auto response = client.Post("/v1/chat/completions", {{"Origin", "https://app"}}, "{}",
                                    "application/json")) {
        failures += check(response->get_header_value("Access-Control-Allow-Origin") == "*",
                          "POST response lacks Access-Control-Allow-Origin");
        failures += check(response->get_header_value("Access-Control-Expose-Headers") ==
                              "x-request-id, request-id",
                          "POST response does not expose the request id headers");
    } else {
        failures += check(false, "POST request failed");
    }
    if (auto response =
            client.Post("/v1/stream", {{"Origin", "https://app"}}, "{}", "application/json")) {
        failures += check(response->status == 200, "streamed response is not 200");
        failures += check(response->get_header_value("Access-Control-Allow-Origin") == "*",
                          "streamed response lacks Access-Control-Allow-Origin");
    } else {
        failures += check(false, "streamed request failed");
    }
    if (auto response = client.Get("/no/such/route")) {
        failures += check(response->get_header_value("Access-Control-Allow-Origin") == "*",
                          "error response lacks Access-Control-Allow-Origin");
    } else {
        failures += check(false, "error request failed");
    }

    server.stop();
    worker.join();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
