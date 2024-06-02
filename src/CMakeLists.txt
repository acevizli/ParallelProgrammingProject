Cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(benchmark REQUIRED)


find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
cuda_add_library(kernels cuda_kernels.cu)
target_include_directories(kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_FILES
        main.cpp
        matrix_utils.cpp
        ryser_algorithms.cpp

)
function(set_common_properties target)
    target_link_libraries(${target} kernels ${CUDA_LIBRARIES})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${target} benchmark::benchmark)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()


add_executable(RyserBenchmark ${SOURCE_FILES})
target_compile_definitions(RyserBenchmark PRIVATE BENCHMARK_TEST)
set_common_properties(RyserBenchmark)

add_executable(RyserGPU ${SOURCE_FILES})
target_compile_definitions(RyserGPU PRIVATE GPU_TEST)
set_common_properties(RyserGPU)

add_executable(RyserMultiGPU ${SOURCE_FILES})
target_compile_definitions(RyserMultiGPU PRIVATE MULTI_GPU_TEST)
set_common_properties(RyserMultiGPU)

add_executable(RyserCPUPar ${SOURCE_FILES})
target_compile_definitions(RyserCPUPar PRIVATE CPU_PAR_TEST)
set_common_properties(RyserCPUPar)

add_executable(RyserCPU ${SOURCE_FILES})
target_compile_definitions(RyserCPU PRIVATE CPU_TEST)
set_common_properties(RyserCPU)


