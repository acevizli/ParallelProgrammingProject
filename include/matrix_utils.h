#pragma once
#include <vector>
#include <cmath>
#include <array>
#include <cstddef> // For std::size_t
#include <algorithm>
#include <numeric> 
#include <bitset>
constexpr long long pow2(int exponent) {
    return exponent == 0 ? 1 : 2 * pow2(exponent - 1);
}

template<std::size_t... Indices>
constexpr auto makePowersOf2(std::index_sequence<Indices...>) {
    return std::array<long long, sizeof...(Indices)>{pow2(Indices)...};
}

constexpr auto power2 = makePowersOf2(std::make_index_sequence<63>{});
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