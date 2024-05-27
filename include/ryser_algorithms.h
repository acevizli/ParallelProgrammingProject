#pragma once
#include <vector>
#include "matrix_utils.h"

constexpr long long pow2(int exponent) {
    return exponent == 0 ? 1 : 2 * pow2(exponent - 1);
}

template<std::size_t... Indices>
constexpr auto makePowersOf2(std::index_sequence<Indices...>) {
    return std::array<long long, sizeof...(Indices)>{pow2(Indices)...};
}

constexpr auto power2 = makePowersOf2(std::make_index_sequence<63>{});




double computePermanentRyserGreyCodeSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);

double computePermanentRyserGreyCode(double* A, int n);

long double computePermanentRyser(double* A, int n);

double computePermanentRyserPar(double* A, int n);

double computePermanentRyserSparsePar(const std::vector<NonZeroElement>& nonZeroElements, int n);


double computePermanentRyserSparse(const std::vector<NonZeroElement>& nonZeroElements, int n);




double computePermanentSpaRyser(int n, int* crs_ptrs, int* crs_colids, double* crs_values, int* ccs_ptrs, int* ccs_rowids, double* ccs_values);