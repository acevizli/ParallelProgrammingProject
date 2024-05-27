Cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(benchmark REQUIRED)

add_executable(RyserBenchmark
        main.cpp
        matrix_utils.cpp
        ryser_algorithms.cpp
)

target_include_directories(RyserBenchmark PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(RyserBenchmark benchmark::benchmark)
