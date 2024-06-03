#pragma once
#include <vector>
#include <cmath>
#include <array>
#include <cstddef> // For std::size_t
#include <algorithm>
#include <numeric> 
#include <bitset>

#ifdef LONG_TYPE
#warning "USING LONG TYPE"
using value = long long int;
#else
#warning "USING DOUBLE TYPE"
using value = double;
#endif



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

std::vector<NonZeroElement> convertToNonZeroElements(value* A, int n);

value * generateMatrixFlatten(int n,double density);

std::vector<std::vector<value>> generateMatrix(int n,double density);


value* flattenVector(const std::vector<std::vector<value>>& matrix);

void convertToCRS(value* A, int n, int* crs_ptrs, int* crs_colids, value* crs_values);

void convertToCCS(value* A, int n, int* ccs_ptrs, int* ccs_rowids, value* ccs_values);