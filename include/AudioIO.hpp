#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Lightweight WAV loader & preprocessing front-end.
class AudioIO {
public:
    struct FrameBuffer {
        std::vector<float> data;   // concatenated frames (row-major)
        std::size_t frameLength;   // samples per frame
        std::size_t hopLength;     // hop in samples
        std::size_t numFrames() const { return frameLength == 0 ? 0 : data.size() / frameLength; }
    };

    explicit AudioIO(uint32_t target_sr = 16000, float pre_emph = 0.97f);

    // Load a WAV file and return mono PCM (normalized to [-1, 1]).
    // Automatically converts to target sample-rate if needed.
    std::vector<float> loadWavMono(const std::string &path) const;

    // Apply pre-emphasis in-place.
    void preEmphasize(std::vector<float> &signal) const;

    // Produce overlapping frames (Hamming-windowed). Returned buffer is flattened.
    FrameBuffer framing(const std::vector<float> &signal, std::size_t frameLen = 400, std::size_t hop = 160) const;

    // Simple RMS energy (per frame). Result length == numFrames.
    std::vector<float> frameEnergy(const FrameBuffer &frames) const;

    // Remove frames whose RMS (dBFS) < threshold (e.g. −40 dB).
    FrameBuffer applyVAD(const FrameBuffer &frames, float dbThreshold = -40.0f) const;

private:
    uint32_t targetSr_;
    float preEmph_;

    // Internal helpers
    static std::vector<int16_t> readPCM16(const std::string &path, uint32_t &sampleRate, uint16_t &numChannels);
    static std::vector<float> linearResample(const std::vector<float> &input, uint32_t srcRate, uint32_t dstRate);
}; 