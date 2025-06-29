// Fusion.cpp
// Placeholder for gated cross-attention fusion logic.
#include <iostream>
#include "Fusion.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
inline float softmax_norm(const std::vector<float> &logits, std::size_t idx) {
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float denom = 0.0f;
    for (float v : logits) denom += std::exp(v - maxLogit);
    return std::exp(logits[idx] - maxLogit) / denom;
}
} // namespace 

Fusion::Fusion(std::size_t dim, unsigned seed) : dim_(dim) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);

    Wg_.resize(3 * 3 * dim_); // 3 x (3*dim)
    for (float &v : Wg_) v = dist(rng);

    b_.resize(3);
    for (float &v : b_) v = dist(rng);
}

std::vector<float> Fusion::fuse(const std::vector<float> &e1,
                                const std::vector<float> &e2,
                                const std::vector<float> &e3) const {
    if (e1.size() != dim_ || e2.size() != dim_ || e3.size() != dim_) {
        throw std::runtime_error("Fusion: embedding dimension mismatch");
    }

    // Concatenate
    std::vector<float> concat; concat.reserve(3 * dim_);
    concat.insert(concat.end(), e1.begin(), e1.end());
    concat.insert(concat.end(), e2.begin(), e2.end());
    concat.insert(concat.end(), e3.begin(), e3.end());

    // Linear map: z = Wg * concat + b   (Wg shape 3 x (3*dim))
    float z[3] = {b_[0], b_[1], b_[2]};
    for (int row = 0; row < 3; ++row) {
        const float *wRow = Wg_.data() + row * 3 * dim_;
        for (std::size_t i = 0; i < concat.size(); ++i) {
            z[row] += wRow[i] * concat[i];
        }
    }

    // Softmax to get gates
    lastGates_.resize(3);
    for (int i = 0; i < 3; ++i) lastGates_[i] = softmax_norm({z[0], z[1], z[2]}, i);

    // Weighted combination
    std::vector<float> fused(dim_, 0.0f);
    for (std::size_t d = 0; d < dim_; ++d) {
        fused[d] = lastGates_[0] * e1[d] + lastGates_[1] * e2[d] + lastGates_[2] * e3[d];
    }
    return fused;
} 