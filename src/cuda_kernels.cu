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
    computePermanentRyserSparseKernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_nonZeroElements, nonZeroCount, n, C, d_results);

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

__global__ void computePermanentSpaRyserGPU(int allThreads, int chunkSize, int n, double* p_vec, double* x, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    extern __shared__ double shared_mem[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= allThreads) return; // Out of bounds check

    int nzeros = 0;
    double prod = 1;

    double inner_p = 0;
    double* inner_x = &shared_mem[threadIdx.x * n];

    // creata a copy of x in the shared mem for each thread
    for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
        //printf("ID: %d, Local: %f, Global: %f\n", int(idx), inner_x[i], x[i]);
    }

    //__syncthreads();

    // define the start and end indices
    unsigned long long start = idx * chunkSize + 1;
    unsigned long long end = ((idx + 1) * chunkSize) + 1 < ((1ULL << n)) ? (((idx + 1) * chunkSize)) + 1 : ((1ULL << n));

    unsigned long long grey_start = (start - 1) ^ ((start - 1) >> 1);


    // calculate inner_x using the grey code before start
    for (unsigned long long i = 0; i < n; i++) { // dim = n
        if ((grey_start >> i) & 0x1) {
            // get the ith col of matrix to sum it w/ x
            for (unsigned long long ptr = ccs_ptrs[i]; ptr < ccs_ptrs[i + 1]; ptr++) { 
                // get the row id and value of each ele in column
                // add it to the corr ele in x
                inner_x[ccs_rowids[ptr]] += ccs_values[ptr];
            }
        }
    }

    //printf("%d, %d\n", int(start), int(end));

    for (unsigned long long i = 0; i < n; i++) { // dim?
        if (inner_x[i] != 0) {
            prod *= inner_x[i];
        } 
        else {
            nzeros++;
        }
    }
    
    for (unsigned long long g = start; g < end; g++) {
  
        unsigned long long grey_prev = (g - 1) ^ ((g - 1) >> 1);
        unsigned long long grey = g ^ (g >> 1);
        unsigned long long diff = grey ^ grey_prev;
        int j = 0;

        while (!(diff & 1)) {
            diff >>= 1;
            j++;
        }

        int s = (grey & (1ULL << j)) ? 1 : -1;

        for (int ptr = ccs_ptrs[j]; ptr < ccs_ptrs[j + 1]; ptr++) {
            int row = ccs_rowids[ptr];
            double val = ccs_values[ptr];

            if (inner_x[row] == 0) {
                nzeros -= 1;
                inner_x[row] += (s * val);
                prod *= inner_x[row];
            } 
            
            else {
                prod /= inner_x[row];
                inner_x[row] += (s * val);

                if (inner_x[row] == 0) {
                    nzeros += 1;
                } 
                else {
                    prod *= inner_x[row];
                }
            }
        }

        if (nzeros == 0) {
            int sign = (g % 2 == 0) ? 1 : -1; 
            inner_p += sign * prod;
            // printf("Thread %llu, g: %llu, sign: %f, prod: %f, inner_p: %f\n", idx, g, sign, prod, inner_p);
        }
    }

    p_vec[idx] = inner_p;
}

double computePermanentSpaRyserMain(int n, int nnz, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    double* x = (double*)malloc(n * sizeof(double));

    int nzeros = 0;
    double p = 1;

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int ptr = crs_ptrs[i]; ptr < crs_ptrs[i + 1]; ptr++) {
            sum += crs_values[ptr];
        }

        x[i] = crs_values[crs_ptrs[i + 1] - 1] - (sum / 2); // one element for each row

        if (x[i] == 0) {
            nzeros++;
        }
    }
 
    if (nzeros == 0) {
        for (int j = 0; j < n; j++) {
            p *= x[j];
        }
    } 
    else {
        p = 0;
    }

    // 268435456ll threads

    uint64_t C = pow(2, n);
    uint64_t blockSize = 128;
    uint64_t chunkSize = 128;
    uint64_t allThreads = (C + chunkSize - 1) / chunkSize;
    uint64_t numBlocks = (allThreads + blockSize - 1) / blockSize;

    printf("Number of threads: %u\n", allThreads);

    size_t sharedMemSize = blockSize * n * sizeof(double); 

    double* p_vec = (double*)malloc(allThreads * sizeof(double));

    int *d_crs_ptrs, *d_crs_colids, *d_ccs_ptrs, *d_ccs_rowids;
    double *d_crs_values, *d_ccs_values, *d_p_vec, *d_x;

    cudaMalloc((void **)&d_crs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_crs_colids, nnz * sizeof(int));
    cudaMalloc((void **)&d_crs_values, nnz * sizeof(double));
    cudaMalloc((void **)&d_ccs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_ccs_rowids, nnz * sizeof(int));
    cudaMalloc((void **)&d_ccs_values, nnz * sizeof(double));
    cudaMalloc((void **)&d_p_vec, allThreads * sizeof(double));
    cudaMalloc((void **)&d_x, n * sizeof(double));

    cudaMemcpy(d_crs_ptrs, crs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_colids, crs_colids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_values, crs_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_ptrs, ccs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_rowids, ccs_rowids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_values, ccs_values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p_vec, p_vec, allThreads * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    computePermanentSpaRyserGPU<<<numBlocks, blockSize, sharedMemSize>>>(allThreads, chunkSize, n, d_p_vec, d_x, d_crs_ptrs, d_crs_colids, d_crs_values, d_ccs_ptrs, d_ccs_rowids, d_ccs_values);
    cudaDeviceSynchronize();

    cudaMemcpy(p_vec, d_p_vec, allThreads * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = p;
#pragma omp parallel for reduction(+: sum)
    for (unsigned long long i = 0; i < allThreads; ++i) {
        sum += p_vec[i];
    }

    sum = -sum * (2 * (n % 2) - 1);
    
    cudaFree(d_p_vec); cudaFree(d_x); cudaFree(d_crs_ptrs);
    cudaFree(d_crs_colids); cudaFree(d_crs_values); cudaFree(d_ccs_ptrs);
    cudaFree(d_ccs_rowids); cudaFree(d_ccs_values);

    free(x);
    free(p_vec);

    return sum;
}