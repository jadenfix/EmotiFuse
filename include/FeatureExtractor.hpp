#pragma once

#include "AudioIO.hpp"
#include <vector>
#include <cstddef>

class FeatureExtractor {
public:
    explicit FeatureExtractor(uint32_t sample_rate = 16000, std::size_t n_mels = 40, std::size_t n_mfcc = 13);

    // Extract 13 MFCC + Δ + ΔΔ = 39 dims per frame.
    // Returns flattened matrix (frames × 39).
    std::vector<float> extract(const AudioIO::FrameBuffer &frames) const;

    std::size_t featureDim() const { return nMfcc_ * 3; }

private:
    uint32_t sr_;
    std::size_t nMels_;
    std::size_t nMfcc_;

    // Precomputed mel filterbank (n_mels × (N/2+1))
    std::vector<std::vector<float>> melFB_;

    void buildMelFilterBank(std::size_t fftSize);

    static float hzToMel(float hz);
    static float melToHz(float mel);
}; 