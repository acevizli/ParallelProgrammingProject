#pragma once
#include "matrix_utils.h"



double computePermanentSpaRyserPar(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);