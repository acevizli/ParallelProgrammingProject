#include "matrix_utils.h"
#include "ryser_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
__global__ void computePermanentRyserSparseKernel(const NonZeroElement* nonZeroElements, int nonZeroCount, int n, unsigned long long C, double* results) {
    extern __shared__ double shared[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= C) return; // Out of bounds check

    unsigned long long k = idx + 1; // Skip k = 0

    double rowsumprod = 1;

    int chi[64] = {0};    

    // Aggregate contributions to row sums from each non-zero element
     int pos = n - 1;

    // note: this will crash if dim < log_2(n)...
    int k2 = k;
    while (k2 > 0)
    {
        chi[pos] = k2 % 2;
        chi[n] += chi[pos];
        k2 = k2 / 2; // integer division        
        pos--;
    }
    double rowSum[64] = {0}; // Max n = 64 due to bitset size
    for (int i = 0; i < nonZeroCount; ++i) {
        if (chi[nonZeroElements[i].col]) {
            rowSum[nonZeroElements[i].row] += nonZeroElements[i].value;
        }
    }

    // Compute the product of the row sums
    for (int i = 0; i < n; ++i) {
        rowsumprod *= rowSum[i];
        if (rowsumprod == 0) break; // Optimization: if product is zero, no need to continue
    }
    int sign = ((n - chi[n]) % 2) ? -1 : 1;
    shared[threadIdx.x] = sign * rowsumprod;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

 double computePermanentRyserSparseCUDA(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    unsigned long long C = 1ULL << n; // 2^n
    int nonZeroCount = nonZeroElements.size();

    // Allocate memory on the device
    NonZeroElement* d_nonZeroElements;
    cudaMalloc(&d_nonZeroElements, nonZeroCount * sizeof(NonZeroElement));
    cudaMemcpy(d_nonZeroElements, nonZeroElements.data(), nonZeroCount * sizeof(NonZeroElement), cudaMemcpyHostToDevice);

    double* d_results;

        int blockSize = 256;
    int numBlocks = (C + blockSize - 1) / blockSize;

    cudaMalloc(&d_results, numBlocks * sizeof(double));

    // Launch the kernel
    computePermanentRyserSparseKernel<<<numBlocks, blockSize,blockSize * sizeof(double)>>>(d_nonZeroElements, nonZeroCount, n, C, d_results);

    // Copy results back to host
    std::vector<double> results(numBlocks);
    cudaMemcpy(results.data(), d_results, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_nonZeroElements);
    cudaFree(d_results);

    // Sum up the results
    double sum = 0;
    for (unsigned long long i = 0; i < numBlocks; ++i) {
        sum += results[i];
    }

    return sum;
}