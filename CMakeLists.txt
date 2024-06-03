cmake_minimum_required(VERSION 3.10)
project(RyserBenchmarkProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the path to the directory containing benchmarkConfig.cmake
set(BENCHMARK_INSTALL_DIR "/home/cs406.ozgunyargi/project/benchmark/build")

# Add the installation prefix of "benchmark" to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${BENCHMARK_INSTALL_DIR})

# Try to find the benchmark package
find_package(benchmark REQUIRED)

if(NOT benchmark_FOUND)
    message(FATAL_ERROR "Could not find Google Benchmark library.")
endif()

add_subdirectory(src)
#add_subdirectory(tests)