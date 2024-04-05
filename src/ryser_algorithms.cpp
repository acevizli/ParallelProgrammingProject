#include "ryser_algorithms.h"
#include "matrix_utils.h"
#include <benchmark/benchmark.h>
#include <omp.h>

long long computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    long long sum = 0;
    std::vector<long long> rowSum(n, 0);
    unsigned long long totalSubsets = power2[n];

    std::bitset<64> chi(0);

    for (unsigned long long k = 1; k < totalSubsets; k++) {
        std::bitset<64> nextChi(k ^ (k >> 1));
        std::bitset<64> diff = chi ^ nextChi;
        int changedIndex = __builtin_ctzll(diff.to_ullong());
        if (nextChi[changedIndex]) {
            for (const auto& elem : nonZeroElements) {
                if (elem.col == changedIndex) {
                    rowSum[elem.row] += elem.value;
                }
            }
        } else {
            for (const auto& elem : nonZeroElements) {
                if (elem.col == changedIndex) {
                    rowSum[elem.row] -= elem.value;
                }
            }
        }

        long long rowSumProd = 1;
        for (int i = 0; i < n; i++) {
            rowSumProd *= rowSum[i];
            if (rowSumProd == 0) break;
        }

        int sign = ((n - nextChi.count()) % 2) ? -1 : 1;
        sum += sign * rowSumProd;

        chi = nextChi;
    }

    return sum;
}

long long computePermanentRyserGreyCode(int* A, int n) {
    long long sum = 0;
    std::vector<long long> rowSum(n, 0); 
    unsigned long long totalSubsets = power2[n]; // Total number of subsets is 2^n
    std::bitset<64> chi(0);
    
    //#pragma omp parallel for reduction(+:sum) num_threads(12)
    for (unsigned long long k = 1; k < totalSubsets; k++) {
        std::bitset<64> nextChi(k ^ (k >> 1));
        // Determine which bit changed
        std::bitset<64> diff = chi ^ nextChi;
        int changedIndex = __builtin_ctzll(diff.to_ullong());

        if (nextChi[changedIndex]) { // If the bit is set in nextChi, the column is added
            for (int i = 0; i < n; i++) {
                rowSum[i] += A[i * n + changedIndex];
            }
        } else { // If the bit is not set in nextChi, the column is removed
            for (int i = 0; i < n; i++) {
                rowSum[i] -= A[i * n + changedIndex];
            }
        }

        // Compute product of row sums for the current subset
        long long rowSumProd = 1;
        for (int i = 0; i < n; i++) {
            rowSumProd *= rowSum[i];
            if (rowSumProd == 0) break; // Early termination if product is zero
        }

        // Update the permanent with the current subset's contribution
        int sign = ((n - nextChi.count()) % 2) ? -1 : 1;
        sum += sign * rowSumProd;
        //benchmark::DoNotOptimize(sum);
        chi = nextChi; // Move to the next subset
    }

    return sum;
}

long long computePermanentRyser(int* A, int n) {
    long sum = 0;
    long rowsumprod, rowsum;   
    unsigned long long C = power2[n]; 

    // loop all 2^n submatrices of A
    //#pragma omp parallel for reduction(+:sum) num_threads(12)
    for (unsigned long long k = 1; k < C; k++)
    {
        rowsumprod = 1;
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        for (int m = 0; m < n; m++)
        {
            rowsum = 0;

            for (int p = 0; p < n; p++)
                rowsum += chi[p] * A[m * n + p];
        

            rowsumprod *= rowsum;    
        
            // (optional -- use for sparse matrices)
            if (rowsumprod == 0) break;    
        }        
        int sign = ((n - chi.count()) % 2) ? -1:1;
        sum += sign * rowsumprod;
        benchmark::DoNotOptimize(sum);

    }   
    //std::cout <<sum<<std::endl;
    //benchmark::DoNotOptimize(sum);
    return sum;
}

long long computePermanentRyserPar(int* A, int n) {
    long sum = 0; 
    unsigned long long C = power2[n]; 
    // loop all 2^n submatrices of A
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long long k = 1; k < C; k++)
    {
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        long rowsumprod = 1, rowsum;  
        for (int m = 0; m < n; m++)
        {
            rowsum = 0;

            for (int p = 0; p < n; p++)
                rowsum += chi[p] * A[m * n + p];
        

            rowsumprod *= rowsum;    
        
            // (optional -- use for sparse matrices)
            if (rowsumprod == 0) break;    
        }        
        int sign = ((n - chi.count()) % 2) ? -1:1;
        sum += sign * rowsumprod;
        benchmark::DoNotOptimize(sum);

    }   
    //std::cout <<sum<<std::endl;
    //benchmark::DoNotOptimize(sum);
    return sum;
}

long long computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    long sum = 0;
    unsigned long long C = power2[n]; 

    // loop all 2^n submatrices of A
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long long k = 1; k < C; k++)
    {
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        std::vector<long long> rowSum(n, 0); // Accumulate row sums here
        long rowsumprod = 1;

        // Aggregate contributions to row sums from each non-zero element
        // that is included in the current subset (chi)
        for (const auto& elem : nonZeroElements) {
            if (chi[elem.col]) {
                rowSum[elem.row] += elem.value;
            }
        }
        // Compute the product of the row sums
        for (int i = 0; i < n; i++) {
            rowsumprod *= rowSum[i];
            if (rowsumprod == 0) break; // Optimization: if product is zero, no need to continue
        }
        int sign = ((n - chi.count()) % 2) ? -1:1;
        sum += sign * rowsumprod;
        benchmark::DoNotOptimize(sum);

    }   
    //std::cout <<sum<<std::endl;
    //benchmark::DoNotOptimize(sum);
    return sum;
}

long long computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    long sum = 0;
    unsigned long long C = power2[n]; 

    // loop all 2^n submatrices of A
    //#pragma omp parallel for reduction(+:sum)
    for (unsigned long long k = 1; k < C; k++)
    {
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        std::vector<long long> rowSum(n, 0); // Accumulate row sums here
        long rowsumprod = 1;

        // Aggregate contributions to row sums from each non-zero element
        // that is included in the current subset (chi)
        for (const auto& elem : nonZeroElements) {
            if (chi[elem.col]) {
                rowSum[elem.row] += elem.value;
            }
        }
        // Compute the product of the row sums
        for (int i = 0; i < n; i++) {
            rowsumprod *= rowSum[i];
            if (rowsumprod == 0) break; // Optimization: if product is zero, no need to continue
        }
        int sign = ((n - chi.count()) % 2) ? -1:1;
        sum += sign * rowsumprod;
        benchmark::DoNotOptimize(sum);

    }   
    //std::cout <<sum<<std::endl;
    //benchmark::DoNotOptimize(sum);
    return sum;
}
