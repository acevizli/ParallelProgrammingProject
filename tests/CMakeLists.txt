cmake_minimum_required(VERSION 3.10)
project(MatrixTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wno-deprecated-copy)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.10.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(benchmark REQUIRED)

add_executable(ryser_tests ryser_tests.cpp ${PROJECT_SOURCE_DIR}/../src/matrix_utils.cpp ${PROJECT_SOURCE_DIR}/../src/ryser_algorithms.cpp)

target_include_directories(ryser_tests PRIVATE ${PROJECT_SOURCE_DIR}/../include)

target_link_libraries(ryser_tests gtest gtest_main benchmark::benchmark)
