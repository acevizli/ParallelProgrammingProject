Cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(benchmark REQUIRED)


find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
cuda_add_library(kernels cuda_kernels.cu)
target_include_directories(kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_executable(RyserBenchmark
        main.cpp
        matrix_utils.cpp
        ryser_algorithms.cpp

)
target_link_libraries(RyserBenchmark kernels ${CUDA_LIBRARIES})
target_include_directories(RyserBenchmark PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(RyserBenchmark benchmark::benchmark)
