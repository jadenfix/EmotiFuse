cmake_minimum_required(VERSION 3.15)
project(emotifuse_cpp LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try to locate pybind11. If not found, the user must install it or add as submodule.
find_package(pybind11 REQUIRED)

# Try to locate ONNX Runtime. Users can set ONNXRUNTIME_DIR to custom path.
find_package(onnxruntime REQUIRED)

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