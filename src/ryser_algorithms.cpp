#include "ryser_algorithms.h"
#include "matrix_utils.h"
#include <benchmark/benchmark.h>
#include <omp.h>
#include <iostream>
double computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    double sum = 0;
    std::vector<double> rowSum(n, 0);
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

        double rowSumProd = 1;
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

double computePermanentRyserGreyCode(double* A, int n) {
    double sum = 0;
    std::vector<double> rowSum(n, 0); 
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
        double rowSumProd = 1;
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

long double computePermanentRyser(double* A, int n) {
    long double sum = 0.0;
    long double rowsumprod, rowsum;   
    unsigned long long C = power2[n]; 

    // loop all 2^n submatrices of A
    //#pragma omp parallel for reduction(+:sum) num_threads(12)
    for (unsigned long long k = 1; k < C; k++)
    {
        rowsumprod = 1.0;
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        for (int m = 0; m < n; m++)
        {
            rowsum = 0.0;

            for (int p = 0; p < n; p++) {
                //if(chi[p]) rowsum+= A[m * n + p];
                rowsum += chi[p] * A[m * n + p];
                //std::cout <<"rowsum: "<< rowsum<<std::endl;
            }
        
            rowsumprod *= rowsum;    


            // (optional -- use for sparse matrices)
            if (rowsumprod == 0.0) break;    
        }        
        int sign = ((n - chi.count()) % 2) ? -1:1;
        sum+= sign * rowsumprod;
                //std::cout <<"sun: "<< sum<<" rowsumprod: "<<rowsumprod<<std::endl;

        benchmark::DoNotOptimize(sum);

    }   
    //std::cout <<sum<<std::endl;
    //benchmark::DoNotOptimize(sum);
    return sum;
}

double computePermanentRyserPar(double* A, int n) {
    double sum = 0; 
    unsigned long long C = power2[n]; 
    // loop all 2^n submatrices of A
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long long k = 1; k < C; k++)
    {
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        double rowsumprod = 1, rowsum;  
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

double computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n) {
    double sum = 0;
    unsigned long long C = power2[n]; 

    // loop all 2^n submatrices of A
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long long k = 1; k < C; k++)
    {
        std::bitset<64> chi(k);
        //benchmark::DoNotOptimize(chi);
        //int * chi = dec2binarr(k,n);
        // loop columns of submatrix #k
        std::vector<double> rowSum(n, 0); // Accumulate row sums here
        double rowsumprod = 1;

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

double computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n) {
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
        std::vector<double> rowSum(n, 0); // Accumulate row sums here
        double rowsumprod = 1;

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





double computePermanentSpaRyser(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {
    double* x = (double*)malloc(n * sizeof(double));

    int nzeros = 0;
    //#pragma omp parallel for reduction(+:nzeros)
    for (int i = 0; i < n; i++){
        double sum = 0.0;
        for (int ptr = crs_ptrs[i]; ptr < crs_ptrs[i+1]; ptr++){
            sum = sum + crs_values[ptr];
        }
        x[i] = crs_values[crs_ptrs[i+1] - 1] - (sum / 2);

        if (x[i] == 0){
            nzeros = nzeros + 1;
        }
    }

    double p = 1.0;
    
    if (nzeros == 0){
        for (int j = 0; j < n; j++){
            p = p * x[j];
        }
    }
    else {
        p = 0.0;
    }


    int ctr = 0;

    for (unsigned long long g = 1; g < power2[n]; g++){
        unsigned long long grey_prev((g-1) ^((g-1) >> 1));
         unsigned long long grey(g ^(g >> 1));
        unsigned long long diff = grey ^ grey_prev;
        int j = 0;
        while (!(diff & 1)) {
            diff >>= 1;
            j++;
        }

        int s = (grey & (1ULL << j)) ? 1 : -1;

        for (int ptr = ccs_ptrs[j]; ptr < ccs_ptrs[j+1]; ptr++){
            int row = ccs_rowids[ptr];
            double val = ccs_values[ptr];

            if (x[row] == 0.0){
                nzeros = nzeros - 1;
            }

            x[row] = x[row] + (s * val);

            if (x[row] == 0.0){
                nzeros = nzeros + 1;
            }
        }

        if (nzeros == 0) {
            ctr++;
            double prod = 1.0;
            //#pragma omp parallel for reduction(*:prod)
            for (int i = 0; i < n; i++) {
                prod = prod * x[i];
            }
            auto sign = (g%2)? -1 : 1;
            auto val = sign * prod;

            p += val;
        } 
        grey_prev = grey;    
    }

    return -p * (2 * (n % 2) - 1);
}



double computePermanentSpaRyserPar(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {

    double* x = (double*)malloc(n * sizeof(double));

    int nzeros = 0;
    double p = 1;

#pragma omp parallel for reduction(+: nzeros) num_threads(16)
    for (int i = 0; i < n; i++) {
        double sum = 0;
        #pragma omp parallel for reduction(+: sum) num_threads(16)
        for (int ptr = crs_ptrs[i]; ptr < crs_ptrs[i + 1]; ptr++) {
            sum += crs_values[ptr];
        }

        x[i] = crs_values[crs_ptrs[i + 1] - 1] - (sum / 2); // one element for each row

        if (x[i] == 0) {
            nzeros++;
        }
    }
 
    if (nzeros == 0) {
#pragma omp parallel for reduction(*: p) num_threads(16)
        for (int j = 0; j < n; j++) {
            p *= x[j];
        }
    } 
    else {
        p = 0;
    }

    int nthrds = 16;

    unsigned long long chunkSize = (1ULL << n) / nthrds;

#pragma omp parallel num_threads(16)
{
    int tid = omp_get_thread_num();

    nzeros = 0;
    double prod = 1;

    double inner_p = 0;
    double * inner_x = (double *)alloca(n * sizeof(double)); 

    // creata a copy of x in the shared mem for each thread

   for (unsigned long long i = 0; i < n; i++){
        inner_x[i] = x[i];
    }


    // define the start and end indices
    unsigned long long start = tid * chunkSize + 1;
    unsigned long long end = ((tid + 1) * chunkSize) + 1 < ((1ULL << n)) ? (((tid + 1) * chunkSize)) + 1 : ((1ULL << n));

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
    #pragma omp critical
        p += inner_p;
    }

    return -p * (2 * (n % 2) - 1);
}
