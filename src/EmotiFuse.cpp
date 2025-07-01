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
        size_t featDim = featExtr_.featureDim();
        
        // MLP branch outputs [numFrames, featDim], but we need [1, featDim] for fusion
        // So we reshape the input to just one frame worth of features (take the mean)
        size_t numFrames = frames.numFrames();
        std::vector<float> meanFeats(featDim, 0.0f);
        for (size_t i = 0; i < numFrames; ++i) {
            for (size_t j = 0; j < featDim; ++j) {
                meanFeats[j] += feats[i * featDim + j];
            }
        }
        for (size_t j = 0; j < featDim; ++j) {
            meanFeats[j] /= numFrames;
        }
        
        std::vector<float> mlpOut = mlpBranch_->run({1, static_cast<int64_t>(featDim)}, meanFeats);

        // Wav2vec & spec transformer branches – dummy zeros (identity models in tests)
        std::vector<float> wvOut = wav2vecBranch_->run({1, static_cast<int64_t>(featDim)}, std::vector<float>(featDim, 0.0f));
        std::vector<float> specOut = specBranch_->run({1, static_cast<int64_t>(featDim)}, std::vector<float>(featDim, 0.0f));

        // All outputs should now have size featDim
        std::cout << "[EmotiFuse] mlpOut size: " << mlpOut.size() << ", wvOut size: " << wvOut.size() << ", specOut size: " << specOut.size() << std::endl;

        // Fuse (use feature dimension)
        fusion_ = Fusion(featDim);
        auto fused = fusion_.fuse(wvOut, mlpOut, specOut);

        // NCDE (identity in tests)
        auto ncdeOut = ncdeBranch_->run({1, static_cast<int64_t>(featDim)}, fused);

        // Classifier (identity) -> logits
        auto logits = classifierBranch_->run({1, static_cast<int64_t>(featDim)}, ncdeOut);
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