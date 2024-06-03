#include "matrix_utils.h"
#include "ryser_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void computePermanentRyserSparseKernel(const NonZeroElement* nonZeroElements, int nonZeroCount, int n, unsigned long long C, value* results) {
    extern __shared__ value shared[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= C) return; // Out of bounds check

    unsigned long long k = idx + 1; // Skip k = 0

    value rowsumprod = 1;

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
    value rowSum[64] = {0}; // Max n = 64 due to bitset size
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

 value computePermanentRyserSparseCUDA(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    unsigned long long C = 1ULL << n; // 2^n
    int nonZeroCount = nonZeroElements.size();

    // Allocate memory on the device
    NonZeroElement* d_nonZeroElements;
    cudaMalloc(&d_nonZeroElements, nonZeroCount * sizeof(NonZeroElement));
    cudaMemcpy(d_nonZeroElements, nonZeroElements.data(), nonZeroCount * sizeof(NonZeroElement), cudaMemcpyHostToDevice);

    value* d_results;

    int blockSize = 256;
    int numBlocks = (C + blockSize - 1) / blockSize;

    cudaMalloc(&d_results, numBlocks * sizeof(value));

    // Launch the kernel
    computePermanentRyserSparseKernel<<<numBlocks, blockSize, blockSize * sizeof(value)>>>(d_nonZeroElements, nonZeroCount, n, C, d_results);

    // Copy results back to host
    std::vector<value> results(numBlocks);
    cudaMemcpy(results.data(), d_results, numBlocks * sizeof(value), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_nonZeroElements);
    cudaFree(d_results);

    // Sum up the results
    value sum = 0;
    for (unsigned long long i = 0; i < numBlocks; ++i) {
        sum += results[i];
    }

    return sum;
}

__device__ int countTrailingZeros(unsigned long long x) {
    return x ? __ffsll(x) - 1 : 64;
}

__inline__ __device__
value warpReduceSum(value val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__
value blockReduceSum(value val) {
    extern __shared__ value shared[]; // Dynamically allocated shared memory
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all warps to finish

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;

    if (wid == 0) val = warpReduceSum(val); // Final reduce within first warp

    return val;
}

__global__ void sumKernel(value* input, value* output, int n) {
    value sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

value computeSum(value* h_input, int n) {
    value *d_input, *d_output;
    int block_size = 256;
    int gridSize = (n + block_size - 1) / block_size;

    cudaMalloc(&d_input, n * sizeof(value));
    cudaMalloc(&d_output, gridSize * sizeof(value));

    cudaMemcpy(d_input, h_input, n * sizeof(value), cudaMemcpyHostToDevice);

    sumKernel<<<gridSize, block_size, block_size * sizeof(value)>>>(d_input, d_output, n);

    value* h_output = (value*)malloc(gridSize * sizeof(value));
    cudaMemcpy(h_output, d_output, gridSize * sizeof(value), cudaMemcpyDeviceToHost);

    value final_sum = 0.0;
    for (int i = 0; i < gridSize; i++) {
        final_sum += h_output[i];
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return final_sum;
}

__global__ void computePermanentSpaRyserGPU(int allThreads, int chunkSize, int n, value* p_vec, value* x, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values) {
    extern __shared__ value shared_mem[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= allThreads) return; // Out of bounds check

    int nzeros = 0;
    value prod = 1;

    value inner_p = 0;
    value * inner_x = (value *)alloca(n * sizeof(value)); 

    // creata a copy of x in the shared mem for each thread
   for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
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
        int j = countTrailingZeros(diff);

        int s = (grey & (1ULL << j)) ? 1 : -1;

        for (int ptr = ccs_ptrs[j]; ptr < ccs_ptrs[j + 1]; ptr++) {
            int row = ccs_rowids[ptr];
            value val = ccs_values[ptr];

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

    shared_mem[threadIdx.x] = inner_p;

//    if(idx < 8){
//    printf("Thread ID: %llu, Shared Mem: %f\n", idx, shared_mem[threadIdx.x]);
//    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
    //    printf("Thread ID: %d, Shared Mem: %f\n", threadIdx.x, shared_mem[0]);
        p_vec[blockIdx.x] = shared_mem[0];
    //    for(int i = 0; i < 10; i++){
    //        printf("Thread ID: %d, Shared Mem: %f\n", i, shared_mem[i]);
    //    }
    }
}

__global__ void computePermanentSpaRyserMultiGPU(int device_id, int allThreads, int chunkSize, int n, value* p_vec, value* x, int* ccs_ptrs, int* ccs_rowids, value* ccs_values) {
    extern __shared__ value shared_mem[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long multi_dev_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (device_id == 1){
        multi_dev_idx += allThreads;
    }

//    if(idx == 0){
//        printf("Thread ID: %d, Multi Device Index: %llu, Device ID: %d, allThreads: %d\n", idx, multi_dev_idx, device_id, allThreads);
//    }



    if (idx >= allThreads) return; // Out of bounds check

    int nzeros = 0;
    value prod = 1;

    value inner_p = 0;
    value * inner_x = (value *)alloca(n * sizeof(value)); 

    // creata a copy of x in the shared mem for each thread
   for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
    }

    //__syncthreads();

    // define the start and end indices
    unsigned long long start = multi_dev_idx * chunkSize + 1;
    unsigned long long end = ((multi_dev_idx + 1) * chunkSize) + 1 < ((1ULL << n)) ? (((multi_dev_idx + 1) * chunkSize)) + 1 : ((1ULL << n));

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
        int j = countTrailingZeros(diff);

        int s = (grey & (1ULL << j)) ? 1 : -1;

        for (int ptr = ccs_ptrs[j]; ptr < ccs_ptrs[j + 1]; ptr++) {
            int row = ccs_rowids[ptr];
            value val = ccs_values[ptr];

//            if (multi_dev_idx == 4){
//                printf("Multidev ID: %d, ptr: %d, row: %d, val: %f\n", multi_dev_idx, ptr, row, val);
//            }

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
//            if(multi_dev_idx == 4){
//                printf("Thread %llu, g: %llu, sign: %f, prod: %f, inner_p: %f\n", idx, g, sign, prod, inner_p);
//            }
        }
    }

    shared_mem[threadIdx.x] = inner_p;

    __syncthreads();

//    if(threadIdx.x < 4){
//    printf("Multi Dev ID: %llu, Shared Mem: %f\n", multi_dev_idx, shared_mem[threadIdx.x]);
//    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        p_vec[blockIdx.x] = shared_mem[0];
//        printf("Device ID: %d, Multi Dev ID: %llu, Sub-Per: %f\n", device_id, multi_dev_idx, shared_mem[0]);
    }

}

value computePermanentSpaRyserMain(int n, int nnz, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values) {
    value* x = (value*)malloc(n * sizeof(value));

    int nzeros = 0;
    value p = 1;

    #pragma omp parallel for reduction(+: nzeros)
    for (int i = 0; i < n; i++) {
        value sum = 0;
        #pragma omp parallel for reduction(+: sum)
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

    uint64_t C = power2[n];
    uint64_t blockSize = 128; // 128;
    uint64_t chunkSize = 128;
    uint64_t allThreads = (C + chunkSize - 1) / chunkSize;
    uint64_t numBlocks = (allThreads + blockSize - 1) / blockSize;

//    printf("Number of threads: %u\n", allThreads);

    size_t sharedMemSize = blockSize * n * sizeof(value); 

    //value* p_vec = (value*)malloc(allThreads * sizeof(value));

    int *d_crs_ptrs, *d_crs_colids, *d_ccs_ptrs, *d_ccs_rowids;
    value *d_crs_values, *d_ccs_values, *d_p_vec, *d_x;

    cudaMalloc((void **)&d_crs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_crs_colids, nnz * sizeof(int));
    cudaMalloc((void **)&d_crs_values, nnz * sizeof(value));
    cudaMalloc((void **)&d_ccs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_ccs_rowids, nnz * sizeof(int));
    cudaMalloc((void **)&d_ccs_values, nnz * sizeof(value));
    cudaMallocManaged((void **)&d_p_vec, numBlocks * sizeof(value));
    cudaMalloc((void **)&d_x, n * sizeof(value));

    cudaMemcpy(d_crs_ptrs, crs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_colids, crs_colids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_values, crs_values, nnz * sizeof(value), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_ptrs, ccs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_rowids, ccs_rowids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_values, ccs_values, nnz * sizeof(value), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_p_vec, p_vec, allThreads * sizeof(value), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(value), cudaMemcpyHostToDevice);

    computePermanentSpaRyserGPU<<<numBlocks, blockSize, blockSize * sizeof(value)>>>(allThreads, chunkSize, n, d_p_vec, d_x, d_crs_ptrs, d_crs_colids, d_crs_values, d_ccs_ptrs, d_ccs_rowids, d_ccs_values);
    cudaDeviceSynchronize();
    //cudaMemcpy(p_vec, d_p_vec, allThreads * sizeof(value), cudaMemcpyDeviceToHost);

//    for (int i = 0; i < numBlocks; i++) {
//        printf("Block %d: %f\n", i, d_p_vec[i]);
//    }

    value sum = p;
    sum+=computeSum(d_p_vec,numBlocks);

    sum = -sum * (2 * (n % 2) - 1);
    
    cudaFree(d_p_vec); cudaFree(d_x); cudaFree(d_crs_ptrs);
    cudaFree(d_crs_colids); cudaFree(d_crs_values); cudaFree(d_ccs_ptrs);
    cudaFree(d_ccs_rowids); cudaFree(d_ccs_values);

    free(x);
    //free(p_vec);

    return sum;
}

__global__ void testMultiGPU(int size, float *a, float *b, float *c) {
    // Get the device ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread ID: %d Working on val a: %f, Working on val b: %f\n",tid, a[tid], b[tid]);
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

value computePermanentSpaRyserMainMultiGPU(int n, int nnz, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values){

    value result = 0.0;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
//    printf("Number of devices: %d\n", deviceCount);
    if (deviceCount < 2) {
//    printf( "We need at least two compute 1.0 or greater "
//    "devices, but only found %d\n", deviceCount );
        return 0;
    }

    value* x = (value*)malloc(n * sizeof(value));
    value* x2 = (value*)malloc(n * sizeof(value));

    int nzeros = 0;
    value p = 1;

    #pragma omp parallel for reduction(+: nzeros)
    for (int i = 0; i < n; i++) {
        value sum = 0;
        #pragma omp parallel for reduction(+: sum)
        for (int ptr = crs_ptrs[i]; ptr < crs_ptrs[i + 1]; ptr++) {
            sum += crs_values[ptr];
        }

        x[i] = crs_values[crs_ptrs[i + 1] - 1] - (sum / 2); // one element for each row
        x2[i] = crs_values[crs_ptrs[i + 1] - 1] - (sum / 2); // one element for each row

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

    uint64_t C = power2[n];
    uint64_t blockSize = 128;
    uint64_t chunkSize = 128;
    uint64_t allThreads = (C + chunkSize - 1) / chunkSize;
    uint64_t numBlocks = (allThreads + blockSize - 1) / blockSize;

//    printf("Number of threads: %u\n", allThreads);
//    printf("Number of blocks: %u\n", numBlocks);

    struct DataStruct {
        int deviceID;
        uint64_t C;
        uint64_t blockSize;
        uint64_t chunkSize;
        uint64_t allThreads;
        uint64_t numBlocks;

        // Permanent related variables
        value *x;
        value *ccs_values;
        int *ccs_ptrs;
        int *ccs_rowids;
        value *p_vec;
    };

    DataStruct data[2];

    data[0].deviceID = 0;
    data[0].C = power2[n]/2;
    data[0].blockSize = 256;
    data[0].chunkSize = 128;
    data[0].allThreads = (data[0].C + data[0].chunkSize - 1) / data[0].chunkSize;
    data[0].numBlocks = (data[0].allThreads + data[0].blockSize - 1) / data[0].blockSize;
    data[0].x = x;
    data[0].ccs_values = ccs_values;
    data[0].ccs_ptrs = ccs_ptrs;
    data[0].ccs_rowids = ccs_rowids;
    data[0].ccs_values = ccs_values;
    value *p_vec_1 = (value*)malloc(data[0].numBlocks * sizeof(value));
    data[0].p_vec = p_vec_1;

    

    data[1].deviceID = 1;
    data[1].C = power2[n]-data[0].C;
    data[1].blockSize = 256;
    data[1].chunkSize = 128;
    data[1].allThreads = (data[1].C + data[1].chunkSize - 1) / data[1].chunkSize;
    data[1].numBlocks = (data[1].allThreads + data[1].blockSize - 1) / data[1].blockSize;
    data[1].x = x2;
    data[1].ccs_values = ccs_values;
    data[1].ccs_ptrs = ccs_ptrs;
    data[1].ccs_rowids = ccs_rowids;
    data[1].ccs_values = ccs_values;
    value *p_vec_2 = (value*)malloc(data[0].numBlocks * sizeof(value));
    data[1].p_vec = p_vec_2;

//    printf("Device ID: %d, C: %llu, blockSize: %d, chunkSize: %d, allThreads: %d, numBlocks: %d\n", data[0].deviceID, data[0].C, data[0].blockSize, data[0].chunkSize, data[0].allThreads, data[0].numBlocks);
//    printf("Device ID: %d, C: %llu, blockSize: %d, chunkSize: %d, allThreads: %d, numBlocks: %d\n", data[1].deviceID, data[1].C, data[1].blockSize, data[1].chunkSize, data[1].allThreads, data[1].numBlocks);


    #pragma omp parallel for
    for (int i = 0; i<2; i++){
        cudaSetDevice(data[i].deviceID);
        int device;
        cudaError_t status = cudaGetDevice(&device);

        if (status != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
        }

//        std::cout << "Current CUDA device ID: " << device << std::endl;
/*
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dim: (" 
                  << deviceProp.maxThreadsDim[0] << ", " 
                  << deviceProp.maxThreadsDim[1] << ", " 
                  << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" 
                  << deviceProp.maxGridSize[0] << ", " 
                  << deviceProp.maxGridSize[1] << ", " 
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
*/

        int *d_crs_ptrs, *d_crs_colids, *d_ccs_ptrs, *d_ccs_rowids;
        value *d_crs_values, *d_ccs_values, *d_p_vec, *d_x;

        cudaMalloc((void **)&d_crs_ptrs, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_crs_colids, nnz * sizeof(int));
        cudaMalloc((void **)&d_crs_values, nnz * sizeof(value));
        cudaMalloc((void **)&d_ccs_ptrs, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_ccs_rowids, nnz * sizeof(int));
        cudaMalloc((void **)&d_ccs_values, nnz * sizeof(value));
        cudaMallocManaged((void **)&d_p_vec, data[i].numBlocks * sizeof(value));
        cudaMalloc((void **)&d_x, n * sizeof(value));

//        cudaMemcpy(d_crs_ptrs, crs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
//        cudaMemcpy(d_crs_colids, crs_colids, nnz * sizeof(int), cudaMemcpyHostToDevice);
//        cudaMemcpy(d_crs_values, crs_values, nnz * sizeof(value), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccs_ptrs, data[i].ccs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccs_rowids, data[i].ccs_rowids, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccs_values, data[i].ccs_values, nnz * sizeof(value), cudaMemcpyHostToDevice);

        //cudaMemcpy(d_p_vec, p_vec, allThreads * sizeof(value), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, data[i].x, n * sizeof(value), cudaMemcpyHostToDevice);

//        printf("Device ID before entering kernel: %d\n", data[i].deviceID);
        computePermanentSpaRyserMultiGPU<<<data[i].numBlocks, data[i].blockSize, data[i].blockSize * sizeof(value)>>>(data[i].deviceID, data[i].allThreads, data[i].chunkSize,n, d_p_vec, d_x, d_ccs_ptrs, d_ccs_rowids, d_ccs_values);
        cudaMemcpy(data[i].p_vec, d_p_vec, data[i].numBlocks * sizeof(value), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

//        for (int i =0; i<data[i].numBlocks; i++){
//            printf("Device ID: %d, Block %d: %f\n", device, i, data[i].p_vec[i]);
//        }
        value sum = p;
//        sum+=computeSum(d_p_vec,numBlocks);

//        sum = -sum * (2 * (n % 2) - 1);
//        res += sum;
//        printf("Sum: %f\n", sum);
        
        cudaFree(d_p_vec); cudaFree(d_x); cudaFree(d_crs_ptrs);
        cudaFree(d_crs_colids); cudaFree(d_crs_values); cudaFree(d_ccs_ptrs);
        cudaFree(d_ccs_rowids); cudaFree(d_ccs_values);

        free(x);
    }

//    for (int i=0; i<data[0].numBlocks; i++){
//        printf("DeviceID: %d, Block %d: %f\n", data[0].deviceID, i, data[0].p_vec[i]);
//    }
//    for (int i=0; i<data[1].numBlocks; i++){
//        printf("DeviceID: %d, Block %d: %f\n", data[1].deviceID, i, data[1].p_vec[i]);
//    }

    value *h_vec = (value*)malloc((data[0].numBlocks+data[1].numBlocks) * sizeof(value));

    memcpy(h_vec, data[0].p_vec, data[0].numBlocks * sizeof(value));
    memcpy(h_vec + data[0].numBlocks, data[1].p_vec, data[1].numBlocks * sizeof(value));

//    for(int i = 0; i < data[0].numBlocks+data[1].numBlocks; i++){
//        printf("Block %d: %f\n", i, h_vec[i]);
//    }
    value sum = p;
    sum += computeSum(h_vec, data[0].numBlocks+data[1].numBlocks);
    sum = -sum * (2 * (n % 2) - 1);

//    printf("Sum: %f\n", sum);

    return sum; //-res * (2 * (n % 2) - 1);
}


/*

    float res = 0.0;
    // Create threads for each GPU
    #pragma omp parallel for reduction(+: res)
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(data[i].deviceID);
        int device;
        cudaError_t status = cudaGetDevice(&device);

        if (status != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
        }

        std::cout << "Current CUDA device ID: " << device << std::endl;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dim: (" 
                  << deviceProp.maxThreadsDim[0] << ", " 
                  << deviceProp.maxThreadsDim[1] << ", " 
                  << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" 
                  << deviceProp.maxGridSize[0] << ", " 
                  << deviceProp.maxGridSize[1] << ", " 
                  << deviceProp.maxGridSize[2] << ")" << std::endl;

        
        int size = data[i].size;
        float *a, *b, *partial_c;
        float *dev_a, *dev_b, *dev_partial_c;

        a = (float*)malloc(size * sizeof(float));
        b = (float*)malloc(size * sizeof(float));
        partial_c = (float*)malloc(size * sizeof(float));

        memcpy(a, data[i].a, size * sizeof(float));
        memcpy(b, data[i].b, size * sizeof(float));

        cudaMalloc((void**)&dev_a, size * sizeof(float));
        cudaMalloc((void**)&dev_b, size * sizeof(float));
        cudaMalloc((void**)&dev_partial_c, size * sizeof(float));

        cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

        testMultiGPU<<<1, size>>>(size, dev_a, dev_b, dev_partial_c);

        cudaMemcpy(partial_c, dev_partial_c, size * sizeof(float), cudaMemcpyDeviceToHost);

        for(int i = 0; i < size; i++) {
            res += partial_c[i];
        }

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_partial_c);

    }
    printf("Result: %f\n", res);

    */