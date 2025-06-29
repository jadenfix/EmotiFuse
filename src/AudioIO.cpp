// AudioIO.cpp
#include "AudioIO.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
constexpr float PI = 3.14159265358979323846f;

inline float hamming(std::size_t n, std::size_t N) {
    return 0.54f - 0.46f * std::cos(2.0f * PI * static_cast<float>(n) / (N - 1));
}

} // namespace

AudioIO::AudioIO(uint32_t target_sr, float pre_emph) : targetSr_(target_sr), preEmph_(pre_emph) {}

// -------------------------- WAV LOADING --------------------------

// Very small PCM16 WAV reader (little-endian)
std::vector<int16_t> AudioIO::readPCM16(const std::string &path, uint32_t &sr, uint16_t &channels) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("AudioIO: cannot open file " + path);
    }

    auto read_u32 = [&f]() {
        uint32_t v;
        f.read(reinterpret_cast<char *>(&v), 4);
        return v;
    };
    auto read_u16 = [&f]() {
        uint16_t v;
        f.read(reinterpret_cast<char *>(&v), 2);
        return v;
    };

    char riff[4];
    f.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        throw std::runtime_error("AudioIO: not a RIFF file");
    }
    read_u32();                 // chunk size
    char wave[4];
    f.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        throw std::runtime_error("AudioIO: not WAVE format");
    }

    // Parse chunks until we hit 'fmt ' and 'data'
    bool fmtFound = false;
    bool dataFound = false;
    uint16_t audioFormat = 0;
    uint32_t dataBytes = 0;
    std::vector<char> dataBuf;

    while (!dataFound && f) {
        char id[4];
        f.read(id, 4);
        uint32_t chunkSize = read_u32();

        if (std::strncmp(id, "fmt ", 4) == 0) {
            audioFormat = read_u16();      // PCM = 1
            channels = read_u16();
            sr = read_u32();
            read_u32();                    // byteRate
            read_u16();                    // blockAlign
            uint16_t bitsPerSample = read_u16();
            if (bitsPerSample != 16) {
                throw std::runtime_error("AudioIO: only 16-bit PCM supported");
            }
            // Skip any extra bytes
            if (chunkSize > 16) {
                f.ignore(chunkSize - 16);
            }
            fmtFound = true;
        } else if (std::strncmp(id, "data", 4) == 0) {
            dataBytes = chunkSize;
            dataBuf.resize(dataBytes);
            f.read(dataBuf.data(), dataBytes);
            dataFound = true;
        } else {
            // skip other chunks
            f.ignore(chunkSize);
        }
    }

    if (!fmtFound || !dataFound) {
        throw std::runtime_error("AudioIO: malformed WAV file");
    }
    if (audioFormat != 1) {
        throw std::runtime_error("AudioIO: unsupported WAV compression (only PCM) ");
    }

    // Copy into vector<int16_t>
    std::vector<int16_t> pcm(dataBytes / 2);
    std::memcpy(pcm.data(), dataBuf.data(), dataBytes);
    return pcm;
}

// Resample via linear interpolation (very small & decent for 16-kHz speech)
std::vector<float> AudioIO::linearResample(const std::vector<float> &input, uint32_t srcRate, uint32_t dstRate) {
    if (srcRate == dstRate) {
        return input;
    }
    double ratio = static_cast<double>(dstRate) / static_cast<double>(srcRate);
    std::size_t outLen = static_cast<std::size_t>(std::round(input.size() * ratio));
    std::vector<float> out(outLen);
    for (std::size_t i = 0; i < outLen; ++i) {
        double srcPos = static_cast<double>(i) / ratio;
        std::size_t idx = static_cast<std::size_t>(srcPos);
        double frac = srcPos - static_cast<double>(idx);
        float s0 = input[idx];
        float s1 = (idx + 1 < input.size()) ? input[idx + 1] : input[idx];
        out[i] = static_cast<float>(s0 + (s1 - s0) * frac);
    }
    return out;
}

std::vector<float> AudioIO::loadWavMono(const std::string &path) const {
    uint32_t sr = 0;
    uint16_t channels = 0;
    auto pcm16 = readPCM16(path, sr, channels);

    // Deinterleave + normalise
    std::vector<float> mono; mono.reserve(pcm16.size() / channels);
    for (std::size_t i = 0; i < pcm16.size(); i += channels) {
        int32_t sum = 0;
        for (uint16_t ch = 0; ch < channels; ++ch) sum += pcm16[i + ch];
        float sample = static_cast<float>(sum) / (channels * 32768.0f);
        mono.push_back(sample);
    }

    // Resample if needed
    if (sr != targetSr_) {
        mono = linearResample(mono, sr, targetSr_);
    }

    // Pre-emphasis
    preEmphasize(mono);
    return mono;
}

void AudioIO::preEmphasize(std::vector<float> &signal) const {
    if (signal.empty()) return;
    for (std::size_t i = signal.size() - 1; i > 0; --i) {
        signal[i] = signal[i] - preEmph_ * signal[i - 1];
    }
}

AudioIO::FrameBuffer AudioIO::framing(const std::vector<float> &signal, std::size_t frameLen, std::size_t hop) const {
    if (signal.size() < frameLen) {
        return {{}, frameLen, hop};
    }
    std::size_t nFrames = 1 + (signal.size() - frameLen) / hop;
    std::vector<float> out(nFrames * frameLen);

    for (std::size_t f = 0; f < nFrames; ++f) {
        const std::size_t offset = f * hop;
        for (std::size_t n = 0; n < frameLen; ++n) {
            float win = hamming(n, frameLen);
            out[f * frameLen + n] = signal[offset + n] * win;
        }
    }
    return {std::move(out), frameLen, hop};
}

std::vector<float> AudioIO::frameEnergy(const FrameBuffer &frames) const {
    std::size_t nFrames = frames.numFrames();
    std::vector<float> energy(nFrames);

    for (std::size_t f = 0; f < nFrames; ++f) {
        const float *ptr = frames.data.data() + f * frames.frameLength;
        float sumSq = 0.0f;
        for (std::size_t n = 0; n < frames.frameLength; ++n) {
            sumSq += ptr[n] * ptr[n];
        }
        float rms = std::sqrt(sumSq / static_cast<float>(frames.frameLength));
        energy[f] = 20.0f * std::log10(std::max(rms, 1e-12f));
    }
    return energy;
}

AudioIO::FrameBuffer AudioIO::applyVAD(const FrameBuffer &frames, float dbThreshold) const {
    auto energy = frameEnergy(frames);
    std::size_t keepCount = 0;
    for (float e : energy) if (e > dbThreshold) ++keepCount;
    if (keepCount == energy.size()) return frames; // nothing removed

    std::vector<float> newData; newData.reserve(keepCount * frames.frameLength);
    for (std::size_t f = 0; f < energy.size(); ++f) {
        if (energy[f] > dbThreshold) {
            const float *src = frames.data.data() + f * frames.frameLength;
            newData.insert(newData.end(), src, src + frames.frameLength);
        }
    }
    return {std::move(newData), frames.frameLength, frames.hopLength};
} 