#pragma once

#include <string>

// Primary entry point to the C++ inference core.
// This scaffold does *not* implement the full pipeline yet; it merely
// exposes a minimal interface so that the Python module can be imported
// and unit tests can be written while development progresses.
class EmotiFuse {
public:
    // Construct with a directory containing the ONNX models. Future
    // implementations will load the required sessions here.
    explicit EmotiFuse(const std::string &model_dir);

    // Predict the emotion label for the given WAV file.
    // TODO: replace std::string return type with a richer result type.
    [[nodiscard]] std::string predict(const std::string &wav_path) const;

private:
    std::string modelDir_;
}; 