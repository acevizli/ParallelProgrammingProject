Cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")


find_package(CUDA REQUIRED)

cuda_add_library(kernels_double cuda_kernels.cu)
target_include_directories(kernels_double PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(kernels_double PROPERTIES CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

cuda_add_library(kernels_long cuda_kernels.cu)
target_include_directories(kernels_long PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(kernels_long PROPERTIES CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
target_compile_definitions(kernels_long PRIVATE LONG_TYPE)

set(SOURCE_FILES
        main.cpp
        matrix_utils.cpp
        ryser_algorithms.cpp

)
function(set_common_properties target)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()



add_executable(RyserGPU_LONG ${SOURCE_FILES})
target_compile_definitions(RyserGPU_LONG PRIVATE GPU_TEST LONG_TYPE)
target_link_libraries(RyserGPU_LONG kernels_long ${CUDA_LIBRARIES})
set_common_properties(RyserGPU_LONG)

add_executable(RyserGPU_DOUBLE ${SOURCE_FILES})
target_compile_definitions(RyserGPU_DOUBLE PRIVATE GPU_TEST)
target_link_libraries(RyserGPU_DOUBLE kernels_double ${CUDA_LIBRARIES})
set_common_properties(RyserGPU_DOUBLE)

add_executable(RyserMultiGPU_DOUBLE ${SOURCE_FILES})
target_compile_definitions(RyserMultiGPU_DOUBLE PRIVATE MULTI_GPU_TEST)
target_link_libraries(RyserMultiGPU_DOUBLE kernels_double ${CUDA_LIBRARIES})
set_common_properties(RyserMultiGPU_DOUBLE)

add_executable(RyserMultiGPU_LONG ${SOURCE_FILES})
target_compile_definitions(RyserMultiGPU_LONG PRIVATE MULTI_GPU_TEST)
target_link_libraries(RyserMultiGPU_DOUBLE kernels_long ${CUDA_LIBRARIES})
set_common_properties(RyserMultiGPU_LONG)

add_executable(RyserCPUPar ${SOURCE_FILES})
target_compile_definitions(RyserCPUPar PRIVATE CPU_PAR_TEST)
set_common_properties(RyserCPUPar)


