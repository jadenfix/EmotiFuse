#include "EmotiFuse.hpp"

#include <iostream>

EmotiFuse::EmotiFuse(const std::string &model_dir) : modelDir_(model_dir) {
    // TODO: initialise ONNX Runtime sessions and preprocessing modules here.
    std::cout << "[EmotiFuse] Initialised with model directory: " << model_dir << std::endl;
}

std::string EmotiFuse::predict(const std::string &wav_path) const {
    // TODO: implement full pipeline. For now, return a stub value.
    std::cout << "[EmotiFuse] Received predict call for: " << wav_path << std::endl;
    return "neutral";
} 