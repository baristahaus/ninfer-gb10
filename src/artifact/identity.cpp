#include "artifact/identity.h"
#include <openssl/evp.h>
#include <algorithm>
#include <fstream>
#include <memory>

namespace ninfer::artifact {
namespace {
class Digest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx_{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
    Digest() {
        if (!ctx_ || EVP_DigestInit_ex(ctx_.get(), EVP_sha256(), nullptr) != 1)
            throw ArtifactError("SHA256 initialization failed");
    }

    void add(const void* data, std::size_t size) {
        if (EVP_DigestUpdate(ctx_.get(), data, size) != 1)
            throw ArtifactError("SHA256 update failed");
    }

    std::string finish() {
        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int size = 0;
        if (EVP_DigestFinal_ex(ctx_.get(), digest, &size) != 1)
            throw ArtifactError("SHA256 finalization failed");
        std::string out;
        constexpr char hex[] = "0123456789abcdef";
        for (unsigned int i = 0; i < size; ++i) {
            out += hex[digest[i] >> 4];
            out += hex[digest[i] & 15];
        }
        return out;
    }
};

std::string hash(std::string_view value) {
    Digest d;
    d.add(value.data(), value.size());
    return d.finish();
}
} // namespace

Json encoded_identity(const std::filesystem::path& path,
                      const std::filesystem::path& template_override) {
    Reader reader(path);
    std::vector<std::string> lines, struct_lines;
    std::vector<std::byte> buffer(4ULL << 20);
    std::uint64_t text_parameters = 0;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<TensorObject>(&object);
        if (!tensor) continue;
        std::string shape;
        for (const auto n : tensor->shape) {
            if (!shape.empty()) shape += ',';
            shape += std::to_string(n);
        }
        const auto line =
            tensor->id + "\n" + tensor->format + "/" + tensor->layout + "\n" + shape + "\n";
        struct_lines.push_back(line);
        Digest content;
        for (std::uint64_t offset = 0; offset < tensor->bytes;) {
            const auto count = static_cast<std::size_t>(
                std::min<std::uint64_t>(buffer.size(), tensor->bytes - offset));
            reader.read_into(tensor->offset + offset, std::span(buffer).first(count));
            content.add(buffer.data(), count);
            offset += count;
        }
        lines.push_back(line + content.finish() + "\n");
    }
    for (const auto& [name, binding] : reader.directory().bindings)
        if (name.starts_with("model.language_model.") &&
            name != "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight")
            text_parameters += binding.elements;
    std::sort(lines.begin(), lines.end());
    std::sort(struct_lines.begin(), struct_lines.end());
    Digest full, structure;
    for (const auto& line : lines) full.add(line.data(), line.size());
    for (const auto& line : struct_lines) structure.add(line.data(), line.size());
    std::string config;
    for (const auto& [name, component] : reader.directory().components)
        config += name + "\n" + component.config.dump() + "\n";
    std::string templ;
    if (template_override.empty()) {
        const auto bytes = reader.read_object(reader.find("frontend/chat_template.jinja"));
        templ.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    } else {
        std::ifstream file(template_override, std::ios::binary);
        if (!file) throw ArtifactError("cannot open chat template for identity");
        templ.assign(std::istreambuf_iterator<char>(file), {});
    }
    return {{"identity_domain", "ninfer.v3.encoded_objects/1"},
            {"model_sha", full.finish()},
            {"model_sha_struct", structure.finish()},
            {"config_sha", hash(config)},
            {"template_sha", hash(templ)},
            {"text_parameters_excluding_file_backed_ple", text_parameters},
            {"metadata", reader.directory().metadata}};
}
} // namespace ninfer::artifact
