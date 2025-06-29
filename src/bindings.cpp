#include <pybind11/pybind11.h>
#include "EmotiFuse.hpp"

namespace py = pybind11;

PYBIND11_MODULE(emotifuse, m) {
    m.doc() = "EmotiFuse C++ core exposed to Python via pybind11 (scaffold)";

    py::class_<EmotiFuse>(m, "EmotiFuse")
        .def(py::init<const std::string &>(), py::arg("model_dir"))
        .def("predict", &EmotiFuse::predict, py::arg("wav_path"));
} 