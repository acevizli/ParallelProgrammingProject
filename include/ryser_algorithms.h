#pragma once
#include <vector>
#include "matrix_utils.h"



value computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

value computePermanentRyserGreyCode(value* A, int n);

value computePermanentRyser(value* A, int n);

value computePermanentRyserPar(value* A, int n);

value computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n);


value computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);


value computePermanentSpaRyser(int n, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values);

value computePermanentSpaRyserPar(int n, int* crs_ptrs, int* crs_colids, value* crs_values, int* ccs_ptrs, int* ccs_rowids, value* ccs_values);