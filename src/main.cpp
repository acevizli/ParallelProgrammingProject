
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
static void ryserTestPar(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
    for (auto _ : state) {
        computePermanentRyserPar(matrix,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}

static void ryserTestSparse(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
     auto sparse = convertToNonZeroElements(matrix,state.range(0));
    for (auto _ : state) {
        computePermanentRyserSparse(sparse,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}

static void ryserTestSparseParallel(benchmark::State& state) {
     auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
     auto sparse = convertToNonZeroElements(matrix,state.range(0));
    for (auto _ : state) {
        computePermanentRyserSparsePar(sparse,state.range(0)); // Replace with your naive function call
    }
    delete[] matrix;
}

static void ryserTestSpaRyser(benchmark::State& state) {
    int nnz = static_cast<int>(state.range(0) * state.range(0) * ((double)state.range(1)) / 100.0);

    // CRS
    int* crs_ptrs = (int*)malloc((state.range(0) + 1) * sizeof(int));
    int* crs_colids = (int*)malloc(nnz * sizeof(int));
    double* crs_values = (double*)malloc(nnz * sizeof(double));

    // CCS
    int* ccs_ptrs = (int*)malloc((state.range(0) + 1) * sizeof(int));
    int* ccs_rowids = (int*)malloc(nnz * sizeof(int));
    double* ccs_values = (double*)malloc(nnz * sizeof(double));
    
    auto matrix = generateMatrixFlatten(state.range(0),((double)state.range(1)) / 100.0); // Generate a matrix of size n x n
    
    convertToCRS(matrix, state.range(0), crs_ptrs, crs_colids, crs_values);
    convertToCCS(matrix, state.range(0), ccs_ptrs, ccs_rowids, ccs_values);
    
    for (auto _ : state) {
        computePermanentSpaRyser(state.range(0), crs_ptrs, crs_colids, crs_values, ccs_ptrs, ccs_rowids, ccs_values); // Replace with your naive function call
    }
    delete[] matrix;
}

//BENCHMARK(ryserTest)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
//BENCHMARK(ryserTestPar)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);

BENCHMARK(ryserTestSparse)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
//BENCHMARK(ryserTestGreyCode)->ArgsProduct({benchmark::CreateDenseRange(24,24,4),benchmark::CreateDenseRange(3,3,1)})->Unit(benchmark::kMillisecond);
//BENCHMARK(ryserTestGreyCodeSparse)->ArgsProduct({benchmark::CreateDenseRange(24,24,4),benchmark::CreateDenseRange(3,3,2)})->Unit(benchmark::kMillisecond);

BENCHMARK(ryserTestGreyCodeSparse)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
BENCHMARK(ryserTestSparseParallel)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);
BENCHMARK(ryserTestSpaRyser)->ArgsProduct({benchmark::CreateDenseRange(24,24,1),benchmark::CreateDenseRange(1,10,2)})->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();