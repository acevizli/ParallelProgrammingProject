#pragma once
#include <vector>
#include <cmath>
#include <array>
#include <cstddef> // For std::size_t
#include <algorithm>
#include <numeric> 
#include <bitset>

inline int* dec2binarr(long n, int dim)
{
    // note: res[dim] will save the sum res[0]+...+res[dim-1]
    int* res = (int*)calloc(dim + 1, sizeof(int));   
    int pos = dim - 1;

    // note: this will crash if dim < log_2(n)...
    while (n > 0)
    {
        res[pos] = n % 2;
        res[dim] += res[pos];
        n = n / 2; // integer division        
        pos--;
    }

    return res;
}

struct NonZeroElement {
    int row;
    int col;
    double value;
};

std::vector<NonZeroElement> convertToNonZeroElements(double* A, int n);

double * generateMatrixFlatten(int n,double density);

std::vector<std::vector<double>> generateMatrix(int n,double density);


double* flattenVector(const std::vector<std::vector<double>>& matrix);

void convertToCRS(double* A, int n, int* crs_ptrs, int* crs_colids, double* crs_values);

void convertToCCS(double* A, int n, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);