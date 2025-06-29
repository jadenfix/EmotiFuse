#pragma once

#include <vector>
#include <cstddef>
#include <random>

// Simple gated fusion: produces weights via a learned linear map over concatenated embeddings.
// gates = softmax(Wg * concat(e1,e2,e3) + b)  (3-dimensional)
// fused = gates[0]*e1 + gates[1]*e2 + gates[2]*e3
class Fusion {
public:
    explicit Fusion(std::size_t dim, unsigned seed = 42);

    std::vector<float> fuse(const std::vector<float>& e1,
                            const std::vector<float>& e2,
                            const std::vector<float>& e3) const;

    std::vector<float> lastGates() const { return lastGates_; }

private:
    std::size_t dim_;
    std::vector<float> Wg_; // shape (3, 3*dim)
    std::vector<float> b_;  // length 3

    mutable std::vector<float> lastGates_; // store last softmax for inspection
}; 