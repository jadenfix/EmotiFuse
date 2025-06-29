#include <pybind11/pybind11.h>
#include "EmotiFuse.hpp"
#include "AudioIO.hpp"

namespace py = pybind11;

PYBIND11_MODULE(emotifuse, m) {
    m.doc() = "EmotiFuse C++ core exposed to Python via pybind11 (scaffold)";

    py::class_<EmotiFuse>(m, "EmotiFuse")
        .def(py::init<const std::string &>(), py::arg("model_dir"))
        .def("predict", &EmotiFuse::predict, py::arg("wav_path"));

    py::class_<AudioIO::FrameBuffer>(m, "FrameBuffer")
        .def_property_readonly("frames", [](const AudioIO::FrameBuffer &fb) { return fb.numFrames(); })
        .def_property_readonly("frame_length", &AudioIO::FrameBuffer::frameLength);

    py::class_<AudioIO>(m, "AudioIO")
        .def(py::init<uint32_t, float>(), py::arg("target_sr") = 16000, py::arg("pre_emph") = 0.97f)
        .def("load_wav_mono", &AudioIO::loadWavMono, py::arg("path"))
        .def("framing", &AudioIO::framing, py::arg("signal"), py::arg("frame_len") = 400, py::arg("hop") = 160)
        .def("frame_energy", &AudioIO::frameEnergy)
        .def("apply_vad", &AudioIO::applyVAD, py::arg("frames"), py::arg("db_threshold") = -40.0f);

    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<uint32_t, std::size_t, std::size_t>(),
             py::arg("sample_rate") = 16000, py::arg("n_mels") = 40, py::arg("n_mfcc") = 13)
        .def("extract", &FeatureExtractor::extract, py::arg("frames"))
        .def_property_readonly("feature_dim", &FeatureExtractor::featureDim);

    // BranchONNX wrapper
    py::class_<BranchONNX>(m, "BranchONNX")
        .def(py::init<const std::string &>(), py::arg("model_path"))
        .def("run", [](BranchONNX &b, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            std::vector<int64_t> shape(arr.ndim());
            for (ssize_t i = 0; i < arr.ndim(); ++i) shape[i] = arr.shape(i);
            std::vector<float> input(arr.size());
            std::memcpy(input.data(), arr.data(), arr.nbytes());
            auto out = b.run(shape, input);
            return py::array(out.size(), out.data());
        });
} 