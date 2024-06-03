#include "matrix_utils.h"
#include <cstring>
void convertToCRS(double* A, int n, int* crs_ptrs, int* crs_colids, double* crs_values) {

    //fill the crs_ptrs array
    memset(crs_ptrs, 0, (n + 1) * sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(A[i * n + j] != 0){
                crs_ptrs[i+1]++;
            }
        }
    }

    //now we have cumulative ordering of crs_ptrs.
    for(int i = 1; i <= n; i++) {
        crs_ptrs[i] += crs_ptrs[i-1];
    }

    // we set crs_colids such that for each element, it holds the related column of that element
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(A[i * n + j] != 0){
                int index = crs_ptrs[i];

                crs_colids[index] = j;
                crs_values[index] = A[i * n + j];

                crs_ptrs[i] = crs_ptrs[i] + 1; 
            }
        }
    }

    for(int i = n; i > 0; i--) {
        crs_ptrs[i] = crs_ptrs[i-1];
    }
    
    crs_ptrs[0] = 0;

}


void convertToCCS(double* A, int n, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {

    //fill the crs_ptrs array
    memset(ccs_ptrs, 0, (n + 1) * sizeof(int));
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            if(A[i * n + j] != 0){
                ccs_ptrs[j+1]++;
            }
        }
    }

    //now we have cumulative ordering of crs_ptrs.
    for(int i = 1; i <= n; i++) {
        ccs_ptrs[i] += ccs_ptrs[i-1];
    }

    // we set crs_colids such that for each element, it holds the related column of that element
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            if(A[i * n + j] != 0){
                int index = ccs_ptrs[j];

                ccs_rowids[index] = i;
                ccs_values[index] = A[i * n + j];

                ccs_ptrs[j] = ccs_ptrs[j] + 1; 
            }
        }
    }

    for(int i = n; i > 0; i--) {
        ccs_ptrs[i] = ccs_ptrs[i-1];
    }
    ccs_ptrs[0] = 0;
}
