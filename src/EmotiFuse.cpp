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
    ncdeBranch_    = std::make_unique<BranchONNX>(check("ncde_emoti.onnx"));
    classifierBranch_ = std::make_unique<BranchONNX>(check("classifier.onnx"));

    std::cout << "[EmotiFuse] Loaded branch models from " << model_dir << std::endl;
}

std::string EmotiFuse::predict(const std::string &wav_path) const {
    std::cout << "[EmotiFuse] Predict: " << wav_path << std::endl;
    // For now, we simply verify that audio can be read and models are available.
    try {
        AudioIO aio;
        auto signal = aio.loadWavMono(wav_path);
        auto frames = aio.framing(signal);

        // Classic features for mlp branch
        auto feats = featExtr_.extract(frames);
        std::vector<float> mlpOut = mlpBranch_->run({static_cast<int64_t>(frames.numFrames()), static_cast<int64_t>(featExtr_.featureDim())}, feats);

        // Wav2vec & spec transformer branches – dummy zeros (identity models in tests)
        std::vector<float> wvOut = wav2vecBranch_->run({1, 10}, std::vector<float>(10, 0.0f));
        std::vector<float> specOut = specBranch_->run({1, 10}, std::vector<float>(10, 0.0f));

        // Fuse (assumes same dim 10)
        fusion_ = Fusion(10);
        auto fused = fusion_.fuse(wvOut, mlpOut, specOut);

        // NCDE (identity in tests)
        auto ncdeOut = ncdeBranch_->run({1, 10}, fused);

        // Classifier (identity) -> logits
        auto logits = classifierBranch_->run({1, 10}, ncdeOut);
        // Pick max index as label (stub mapping)
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = i;
        static const char *labels[] = {"neutral", "happy", "sad", "angry", "other"};
        return labels[best % 5];
    } catch (const std::exception &e) {
        std::cerr << "AudioIO error: " << e.what() << std::endl;
    }
    return "error";
} 