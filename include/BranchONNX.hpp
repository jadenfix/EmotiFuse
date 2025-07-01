#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Lightweight wrapper around an ONNX Runtime inference session.
class BranchONNX {
public:
    explicit BranchONNX(const std::string &modelPath);

    // Run inference with a contiguous float input tensor of given shape.
    // Returns output tensor flattened to a std::vector<float>.
    std::vector<float> run(const std::vector<int64_t> &inputShape,
                           const std::vector<float> &input) const;

    std::vector<int64_t> outputShape() const { return outputShape_; }
    
    // Get the actual output shape from the last run() call
    std::vector<int64_t> lastOutputShape() const { return lastOutputShape_; }

private:
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<int64_t> outputShape_;
    mutable std::vector<int64_t> lastOutputShape_;  // mutable so it can be modified in const run()
}; 