#pragma once
#include <vector>
#include "matrix_utils.h"
long long computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

long long computePermanentRyserGreyCode(int* A, int n);

long long computePermanentRyser(int* A, int n);