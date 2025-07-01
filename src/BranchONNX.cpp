// BranchONNX.cpp
#include "BranchONNX.hpp"

#include <stdexcept>
#include <array>

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

    // Get IO names (allocator returns managed strings in >=1.22)
    Ort::AllocatedStringPtr inputNamePtr  = session_.GetInputNameAllocated(0, allocator_);
    Ort::AllocatedStringPtr outputNamePtr = session_.GetOutputNameAllocated(0, allocator_);

    std::array<const char *, 1> inputNames  = {inputNamePtr.get()};
    std::array<const char *, 1> outputNames = {outputNamePtr.get()};

    // Run inference - create non-const copy to call Run
    Ort::Session& mutableSession = const_cast<Ort::Session&>(session_);
    auto outputTensors = mutableSession.Run(Ort::RunOptions{nullptr},
                                           inputNames.data(), &inputTensor, 1,
                                           outputNames.data(), 1);

    float *outPtr = outputTensors[0].GetTensorMutableData<float>();
    auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outShape = typeInfo.GetShape();
    size_t outSize = typeInfo.GetElementCount();

    // Store the actual output shape for later access
    lastOutputShape_ = outShape;

    return std::vector<float>(outPtr, outPtr + outSize);
} 