// BranchONNX.cpp
#include "BranchONNX.hpp"

#include <stdexcept>

namespace {
// Shared environment for all sessions (singleton pattern)
Ort::Env &getEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EmotiFuse");
    return env;
}

Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}
} // namespace

BranchONNX::BranchONNX(const std::string &modelPath) {
    session_ = Ort::Session(getEnv(), modelPath.c_str(), makeSessionOptions());

    // Cache output shape from first output.
    size_t numOut = session_.GetOutputCount();
    if (numOut == 0) {
        throw std::runtime_error("BranchONNX: model has no outputs");
    }
    auto typeInfo = session_.GetOutputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    outputShape_ = tensorInfo.GetShape();
}

std::vector<float> BranchONNX::run(const std::vector<int64_t> &inputShape,
                                   const std::vector<float> &input) const {
    if (!session_) throw std::runtime_error("BranchONNX: session not initialised");

    // Prepare input tensor
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float *>(input.data()), input.size(), inputShape.data(), inputShape.size());

    // Get IO names
    const char *inputName = session_.GetInputName(0, allocator_);
    const char *outputName = session_.GetOutputName(0, allocator_);

    // Run inference
    auto outputTensor = session_.Run(Ort::RunOptions{nullptr},
                                     &inputName, &inputTensor, 1,
                                     &outputName, 1);

    float *outPtr = outputTensor[0].GetTensorMutableData<float>();
    auto typeInfo = outputTensor[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outShape = typeInfo.GetShape();
    size_t outSize = typeInfo.GetElementCount();

    return std::vector<float>(outPtr, outPtr + outSize);
} 