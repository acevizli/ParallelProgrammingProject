#include "ryser_algorithms.h"
#include "matrix_utils.h"
#include <omp.h>

double computePermanentSpaRyserPar(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {

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
#pragma omp parallel for reduction(*: p)
        for (int j = 0; j < n; j++) {
            p *= x[j];
        }
    } 
    else {
        p = 0;
    }

    int nthrds = 16;

    unsigned long long chunkSize = (1ULL << n) / nthrds;

#pragma omp parallel
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
