#pragma once
#include <vector>
#include "matrix_utils.h"






double computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

double computePermanentRyserGreyCode(double* A, int n);

long double computePermanentRyser(double* A, int n);

double computePermanentRyserPar(double* A, int n);

double computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n);


double computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);


double computePermanentSpaRyser(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);

double computePermanentSpaRyserPar(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);