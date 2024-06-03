#include "matrix_utils.h"
value computePermanentRyserSparseCUDA(const std::vector<NonZeroElement>& nonZeroElements, int n);
value computePermanentSpaRyserMain(int n, int nnz, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values);
value computePermanentSpaRyserMainMultiGPU(int n, int nnz, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values);