cmake_minimum_required(VERSION 3.15)
project(emotifuse_cpp LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect the current Python interpreter (needed for pybind11 wheel fallback)
find_package(Python COMPONENTS Interpreter REQUIRED)

# Try to locate pybind11:
#   1. Prefer a system / Homebrew install (find_package CONFIG)
#   2. Otherwise fall back to the path reported by the Python wheel
#      (`python -m pybind11 --cmakedir`). This allows users who simply
#      `pip install pybind11` to build the extension without any extra
#      CMake arguments.

if (NOT TARGET pybind11::pybind11)
    find_package(pybind11 CONFIG QUIET)
    if (NOT pybind11_FOUND)
        message(STATUS "pybind11 not found via CMake; searching via Python wheel…")
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
            RESULT_VARIABLE _pybind11_status
            OUTPUT_VARIABLE _pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (_pybind11_status EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR "pybind11 not found. Please install it (e.g. pip install pybind11) or set pybind11_DIR.")
        endif()
    endif()
endif()

# Try to locate ONNX Runtime
find_package(onnxruntime CONFIG QUIET)

if (NOT onnxruntime_FOUND)
    message(STATUS "onnxruntimeConfig.cmake not found – trying wheel location")
    execute_process(
        COMMAND ${Python_EXECUTABLE} - <<PY
import importlib, sys, pathlib, json, os
try:
    import onnxruntime as ort
except ImportError:
    sys.exit(1)
wheel_dir = pathlib.Path(ort.__file__).parent
inc = wheel_dir / "include"
lib = wheel_dir
print(json.dumps({"inc": str(inc), "lib": str(lib)}))
PY
        RESULT_VARIABLE _ort_status
        OUTPUT_VARIABLE _ort_json
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (_ort_status EQUAL 0)
        file(WRITE ${CMAKE_BINARY_DIR}/ort.json "${_ort_json}")
        file(READ ${CMAKE_BINARY_DIR}/ort.json _ort_content)
        string(JSON _ort_inc GET ${_ort_content} inc)
        string(JSON _ort_libdir GET ${_ort_content} lib)
        message(STATUS "Using ORT wheel include at ${_ort_inc}")
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ort_libdir}/libonnxruntime.dylib")
        target_include_directories(onnxruntime INTERFACE "${_ort_inc}")
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install the SDK or the Python wheel.")
    endif()
endif()

# Include public headers
include_directories(include)

# Build the Python extension module
pybind11_add_module(emotifuse
    src/AudioIO.cpp
    src/FeatureExtractor.cpp
    src/BranchONNX.cpp
    src/Fusion.cpp
    src/NCDE.cpp
    src/EmotiFuse.cpp
    src/bindings.cpp
)

# Link against ONNX Runtime
# The imported target name may vary depending on how onnxruntime was installed.
target_link_libraries(emotifuse PRIVATE onnxruntime::onnxruntime) 