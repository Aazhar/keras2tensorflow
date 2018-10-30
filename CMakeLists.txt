cmake_minimum_required(VERSION 3.5+)
project(keras2tensorflow)
set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/eigen_archive")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/protobuf_archive/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/bazel-genfiles")

add_executable(k2tf main.cpp)

target_link_libraries(k2tf tensorflow_cc tensorflow_framework)