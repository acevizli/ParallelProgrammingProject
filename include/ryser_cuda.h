#include "matrix_utils.h"
double computePermanentRyserSparseCUDA(const std::vector<NonZeroElement>& nonZeroElements, int n);
double computePermanentSpaRyserMain(int n, int nnz, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);
double computePermanentSpaRyserMainMultiGPU(int n, int nnz, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);