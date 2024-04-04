
#include <benchmark/benchmark.h>
#include "matrix_utils.h"
#include "ryser_algorithms.h"

static void ryserTestGreyCodeSparse(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n with density
     auto sparse = convertToNonZeroElements(matrix,state.range(0));
    for (auto _ : state) {
        computePermanentRyserGreyCodeSparse(sparse,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}

static void ryserTestGreyCode(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
    for (auto _ : state) {
        computePermanentRyserGreyCode(matrix,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}

static void ryserTest(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
    for (auto _ : state) {
        computePermanentRyser(matrix,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}


BENCHMARK(ryserTest)->ArgsProduct({benchmark::CreateDenseRange(24,24,4),benchmark::CreateDenseRange(3,3,1)})->Unit(benchmark::kMillisecond);
BENCHMARK(ryserTestGreyCode)->ArgsProduct({benchmark::CreateDenseRange(24,24,4),benchmark::CreateDenseRange(3,3,1)})->Unit(benchmark::kMillisecond);
BENCHMARK(ryserTestGreyCodeSparse)->ArgsProduct({benchmark::CreateDenseRange(24,24,4),benchmark::CreateDenseRange(3,3,2)})->Unit(benchmark::kMillisecond);

BENCHMARK(ryserTestGreyCode)->ArgsProduct({benchmark::CreateDenseRange(28,28,4),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
BENCHMARK(ryserTestGreyCodeSparse)->ArgsProduct({benchmark::CreateDenseRange(28,28,4),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();