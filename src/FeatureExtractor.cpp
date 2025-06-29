// FeatureExtractor.cpp
#include "FeatureExtractor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace {
constexpr float PI = 3.14159265358979323846f;
}

FeatureExtractor::FeatureExtractor(uint32_t sample_rate, std::size_t n_mels, std::size_t n_mfcc)
    : sr_(sample_rate), nMels_(n_mels), nMfcc_(n_mfcc) {
    buildMelFilterBank(512);
}

float FeatureExtractor::hzToMel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
float FeatureExtractor::melToHz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

void FeatureExtractor::buildMelFilterBank(std::size_t fftSize) {
    std::size_t nFftBins = fftSize / 2 + 1;
    float fMin = 0.0f;
    float fMax = static_cast<float>(sr_) / 2.0f;

    float melMin = hzToMel(fMin);
    float melMax = hzToMel(fMax);
    float melStep = (melMax - melMin) / (nMels_ + 1);

    std::vector<float> melCenters(nMels_ + 2);
    for (std::size_t i = 0; i < melCenters.size(); ++i) {
        melCenters[i] = melToHz(melMin + i * melStep);
    }

    melFB_.resize(nMels_, std::vector<float>(nFftBins, 0.0f));

    for (std::size_t m = 0; m < nMels_; ++m) {
        float f_left = melCenters[m];
        float f_center = melCenters[m + 1];
        float f_right = melCenters[m + 2];

        for (std::size_t k = 0; k < nFftBins; ++k) {
            float f_k = static_cast<float>(k) * sr_ / fftSize;
            if (f_k >= f_left && f_k <= f_center) {
                melFB_[m][k] = (f_k - f_left) / (f_center - f_left);
            } else if (f_k > f_center && f_k <= f_right) {
                melFB_[m][k] = (f_right - f_k) / (f_right - f_center);
            }
        }
    }
}

std::vector<float> FeatureExtractor::extract(const AudioIO::FrameBuffer &frames) const {
    const std::size_t fftSize = 512;
    const std::size_t nFftBins = fftSize / 2 + 1;

    std::size_t nFrames = frames.numFrames();
    if (nFrames == 0) return {};

    // Ensure we have filterbank ready
    if (melFB_.empty()) const_cast<FeatureExtractor*>(this)->buildMelFilterBank(fftSize);

    // Buffer to hold log-mel energies per frame
    std::vector<std::vector<float>> melE(nFrames, std::vector<float>(nMels_));

    // Temporary arrays for FFT (naive DFT)
    std::vector<std::complex<float>> X(nFftBins);

    for (std::size_t f = 0; f < nFrames; ++f) {
        const float* framePtr = frames.data.data() + f * frames.frameLength;
        // Zero-pad to fftSize if necessary
        for (std::size_t k = 0; k < nFftBins; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            for (std::size_t n = 0; n < frames.frameLength; ++n) {
                float angle = -2.0f * PI * k * n / fftSize;
                std::complex<float> expTerm(std::cos(angle), std::sin(angle));
                sum += framePtr[n] * expTerm;
            }
            X[k] = sum;
        }
        // Power spectrum
        std::vector<float> power(nFftBins);
        std::transform(X.begin(), X.end(), power.begin(), [](const std::complex<float>& c){ return std::norm(c); });

        // Apply mel filterbank
        for (std::size_t m = 0; m < nMels_; ++m) {
            float melSum = 0.0f;
            for (std::size_t k = 0; k < nFftBins; ++k) {
                melSum += melFB_[m][k] * power[k];
            }
            melE[f][m] = std::log10(std::max(melSum, 1e-12f));
        }
    }

    // DCT-II for MFCC
    std::vector<std::vector<float>> mfcc(nFrames, std::vector<float>(nMfcc_));
    for (std::size_t f = 0; f < nFrames; ++f) {
        for (std::size_t n = 0; n < nMfcc_; ++n) {
            float acc = 0.0f;
            for (std::size_t m = 0; m < nMels_; ++m) {
                acc += melE[f][m] * std::cos(PI * n * (static_cast<float>(m) + 0.5f) / nMels_);
            }
            mfcc[f][n] = acc;
        }
    }

    // Compute deltas (simple diff) and delta-deltas
    std::vector<std::vector<float>> deltas(nFrames, std::vector<float>(nMfcc_, 0.0f));
    for (std::size_t f = 1; f + 1 < nFrames; ++f) {
        for (std::size_t n = 0; n < nMfcc_; ++n) {
            deltas[f][n] = (mfcc[f + 1][n] - mfcc[f - 1][n]) * 0.5f; // scale 1/2
        }
    }

    std::vector<std::vector<float>> delta2(nFrames, std::vector<float>(nMfcc_, 0.0f));
    for (std::size_t f = 1; f + 1 < nFrames; ++f) {
        for (std::size_t n = 0; n < nMfcc_; ++n) {
            delta2[f][n] = (deltas[f + 1][n] - deltas[f - 1][n]) * 0.5f;
        }
    }

    // Flatten to output vector (frame-major)
    std::vector<float> out; out.reserve(nFrames * featureDim());
    for (std::size_t f = 0; f < nFrames; ++f) {
        for (std::size_t n = 0; n < nMfcc_; ++n) out.push_back(mfcc[f][n]);
        for (std::size_t n = 0; n < nMfcc_; ++n) out.push_back(deltas[f][n]);
        for (std::size_t n = 0; n < nMfcc_; ++n) out.push_back(delta2[f][n]);
    }
    return out;
} 