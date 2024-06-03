#include "matrix_utils.h"
#include "ryser_cuda.h"
#include <cuda_runtime.h>
#include <omp.h>

__device__ int countTrailingZeros(unsigned long long x) {
    return x ? __ffsll(x) - 1 : 64;
}

__inline__ __device__
double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__
double blockReduceSum(double val) {
    extern __shared__ double shared[]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); 

    if (lane == 0) shared[wid] = val; 

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void sumKernel(double* input, double* output, int n) {
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

double computeSum(double* h_input, int n) {
    double *d_input, *d_output;
    int block_size = 256;
    int gridSize = (n + block_size - 1) / block_size;

    cudaMalloc(&d_input, n * sizeof(double));
    cudaMalloc(&d_output, gridSize * sizeof(double));

    cudaMemcpy(d_input, h_input, n * sizeof(double), cudaMemcpyHostToDevice);

    sumKernel<<<gridSize, block_size, block_size * sizeof(double)>>>(d_input, d_output, n);

    double* h_output = (double*)malloc(gridSize * sizeof(double));
    cudaMemcpy(h_output, d_output, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    double final_sum = 0.0;
    for (int i = 0; i < gridSize; i++) {
        final_sum += h_output[i];
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return final_sum;
}

__global__ void computePermanentSpaRyserGPU(unsigned long long allThreads, int chunkSize, int n, double* p_vec, double* x, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    extern __shared__ double shared_mem[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= allThreads) return; 

    int nzeros = 0;
    double prod = 1;

    double inner_p = 0;
    double * inner_x = (double *)alloca(n * sizeof(double)); 

   for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
    }
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
        }
    }

    shared_mem[threadIdx.x] = inner_p;


    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        p_vec[blockIdx.x] = shared_mem[0];
    }
}

__global__ void computePermanentSpaRyserMultiGPU(int device_id, unsigned long long allThreads, int chunkSize, int n, double* p_vec, double* x, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    extern __shared__ double shared_mem[];
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long multi_dev_idx = (blockIdx.x * blockDim.x + threadIdx.x)+device_id*allThreads;


    if (idx >= allThreads) return; // Out of bounds check

    int nzeros = 0;
    double prod = 1;

    double inner_p = 0;
    double * inner_x = (double *)alloca(n * sizeof(double)); 

    // creata a copy of x in the shared mem for each thread
   for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
    }

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

        }
    }

    shared_mem[threadIdx.x] = inner_p;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        p_vec[blockIdx.x] = shared_mem[0];
    }

}

double computePermanentSpaRyserMain(int n, int nnz, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    double* x = (double*)malloc(n * sizeof(double));

    int nzeros = 0;
    double p = 1;

    #pragma omp parallel for reduction(+: nzeros)
    for (int i = 0; i < n; i++) {
        double sum = 0;
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
    uint64_t chunkSize = 2048;
    unsigned long long allThreads = (C + chunkSize - 1) / chunkSize;
    uint64_t numBlocks = (allThreads + blockSize - 1) / blockSize;


    size_t sharedMemSize = blockSize * n * sizeof(double); 


    int *d_crs_ptrs, *d_crs_colids, *d_ccs_ptrs, *d_ccs_rowids;
    double *d_crs_values, *d_ccs_values, *d_p_vec, *d_x;

    cudaMalloc((void **)&d_crs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_crs_colids, nnz * sizeof(int));
    cudaMalloc((void **)&d_crs_values, nnz * sizeof(double));
    cudaMalloc((void **)&d_ccs_ptrs, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_ccs_rowids, nnz * sizeof(int));
    cudaMalloc((void **)&d_ccs_values, nnz * sizeof(double));
    cudaMallocManaged((void **)&d_p_vec, numBlocks * sizeof(double));
    cudaMalloc((void **)&d_x, n * sizeof(double));

    cudaMemcpy(d_crs_ptrs, crs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_colids, crs_colids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crs_values, crs_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_ptrs, ccs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_rowids, ccs_rowids, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccs_values, ccs_values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    computePermanentSpaRyserGPU<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(allThreads, chunkSize, n, d_p_vec, d_x, d_crs_ptrs, d_crs_colids, d_crs_values, d_ccs_ptrs, d_ccs_rowids, d_ccs_values);
    cudaDeviceSynchronize();


    double sum = p;
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
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

double computePermanentSpaRyserMainMultiGPU(int n, int nnz, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values){
    int num_gpus = 2;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }


    num_gpus=2;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    double **xs = (double**)malloc(num_gpus * sizeof(double*));

    for (int i = 0; i < num_gpus; i++) {
        xs[i] = (double*)malloc(n * sizeof(double));
    }

    int nzeros = 0;
    double p = 1;
    double *x = (double*)malloc(n * sizeof(double));

    #pragma omp parallel for reduction(+: nzeros)
    for (int i = 0; i < n; i++) {
        double sum = 0;
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

    for (int i = 0; i<num_gpus; i++){
        memcpy(xs[i], x, n * sizeof(double));
    }

    uint64_t C = power2[n];
    uint64_t blockSize = 128;
    uint64_t chunkSize = 128;
    uint64_t allThreads = (C + chunkSize - 1) / chunkSize;
    uint64_t numBlocks = (allThreads + blockSize - 1) / blockSize;

    struct DataStruct {
        int deviceID;
        uint64_t C;
        uint64_t blockSize;
        uint64_t chunkSize;
        uint64_t allThreads;
        uint64_t numBlocks;

        // Permanent related variables
        double *x;
        double *ccs_values;
        int *ccs_ptrs;
        int *ccs_rowids;
        double *p_vec;
    };

    DataStruct *datas = (DataStruct*)malloc(num_gpus * sizeof(DataStruct));
    for (int i = 0; i < num_gpus; i++) {
        datas[i].deviceID = i;
        datas[i].C = power2[n]/num_gpus;
        datas[i].blockSize = 256;
        datas[i].chunkSize = 128;
        datas[i].allThreads = (datas[i].C + datas[i].chunkSize - 1) / datas[i].chunkSize;
        datas[i].numBlocks = (datas[i].allThreads + datas[i].blockSize - 1) / datas[i].blockSize;
        datas[i].x = xs[i];
        datas[i].ccs_values = ccs_values;
        datas[i].ccs_ptrs = ccs_ptrs;
        datas[i].ccs_rowids = ccs_rowids;
        datas[i].ccs_values = ccs_values;
        double *p_vec = (double*)malloc(datas[i].numBlocks * sizeof(double));
        datas[i].p_vec = p_vec;
    }

    uint64_t subset_size = 0;
    for(int i=0; i<num_gpus; i++){
        subset_size += datas[i].C;
    }


    omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);

        int *d_crs_ptrs, *d_crs_colids, *d_ccs_ptrs, *d_ccs_rowids;
        double *d_crs_values, *d_ccs_values, *d_p_vec, *d_x;

        cudaMalloc((void **)&d_crs_ptrs, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_crs_colids, nnz * sizeof(int));
        cudaMalloc((void **)&d_crs_values, nnz * sizeof(double));
        cudaMalloc((void **)&d_ccs_ptrs, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_ccs_rowids, nnz * sizeof(int));
        cudaMalloc((void **)&d_ccs_values, nnz * sizeof(double));
        cudaMallocManaged((void **)&d_p_vec, datas[cpu_thread_id].numBlocks * sizeof(double));
        cudaMalloc((void **)&d_x, n * sizeof(double));

        //##############################################################################################

        cudaMemcpy(d_ccs_ptrs, datas[cpu_thread_id].ccs_ptrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccs_rowids, datas[cpu_thread_id].ccs_rowids, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccs_values, datas[cpu_thread_id].ccs_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, datas[cpu_thread_id].x, n * sizeof(double), cudaMemcpyHostToDevice);

        #pragma omp barrier

        computePermanentSpaRyserMultiGPU<<<datas[cpu_thread_id].numBlocks, datas[cpu_thread_id].blockSize, datas[cpu_thread_id].blockSize * sizeof(double)>>>(datas[cpu_thread_id].deviceID, datas[cpu_thread_id].allThreads, datas[cpu_thread_id].chunkSize,n, d_p_vec, d_x, d_ccs_ptrs, d_ccs_rowids, d_ccs_values);
        cudaMemcpy(datas[cpu_thread_id].p_vec, d_p_vec, datas[cpu_thread_id].numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_p_vec); cudaFree(d_x); cudaFree(d_crs_ptrs);
        cudaFree(d_crs_colids); cudaFree(d_crs_values); cudaFree(d_ccs_ptrs);
        cudaFree(d_ccs_rowids); cudaFree(d_ccs_values);
    }

    int tempNumBlocks = 0;
    for (int i = 0; i < num_gpus; i++) {
        tempNumBlocks += datas[i].numBlocks;
    }

    double *h_vec = (double*)malloc(numBlocks * num_gpus* sizeof(double));

    for (int i = 0; i<num_gpus; i++){
        int gpuNumBlocks = datas[i].numBlocks;
        for (int j = 0; j < gpuNumBlocks; j++) {
            h_vec[j + i * gpuNumBlocks] = datas[i].p_vec[j];
        }
    }

    double sum = p;
    sum += computeSum(h_vec, tempNumBlocks);
    sum = -sum * (2 * (n % 2) - 1);

    return sum;
}