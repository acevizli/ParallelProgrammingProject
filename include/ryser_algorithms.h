#pragma once
#include <vector>
#include "matrix_utils.h"
long long computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

long long computePermanentRyserGreyCode(int* A, int n);

long long computePermanentRyser(int* A, int n);

long long computePermanentRyserPar(int* A, int n);

long long computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n);


long long computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

long long computePermanentSpaRyser(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);