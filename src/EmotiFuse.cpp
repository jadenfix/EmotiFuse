#include "EmotiFuse.hpp"
#include "BranchONNX.hpp"
#include "AudioIO.hpp"

#include <iostream>
#include <filesystem>

EmotiFuse::EmotiFuse(const std::string &model_dir) : modelDir_(model_dir) {
    namespace fs = std::filesystem;
    fs::path dir(model_dir);

    auto check = [&](const char *filename) {
        fs::path p = dir / filename;
        if (!fs::exists(p)) throw std::runtime_error(std::string("Model file not found: ") + p.string());
        return p.string();
    };

    wav2vecBranch_ = std::make_unique<BranchONNX>(check("wav2vec_emoti.onnx"));
    mlpBranch_     = std::make_unique<BranchONNX>(check("mlp_emoti.onnx"));
    specBranch_    = std::make_unique<BranchONNX>(check("spec_transformer.onnx"));

    std::cout << "[EmotiFuse] Loaded branch models from " << model_dir << std::endl;
}

std::string EmotiFuse::predict(const std::string &wav_path) const {
    std::cout << "[EmotiFuse] Predict: " << wav_path << std::endl;
    // For now, we simply verify that audio can be read and models are available.
    try {
        AudioIO aio;
        auto signal = aio.loadWavMono(wav_path);
        (void)signal; // unused for now
    } catch (const std::exception &e) {
        std::cerr << "AudioIO error: " << e.what() << std::endl;
    }
    return "neutral";
} 