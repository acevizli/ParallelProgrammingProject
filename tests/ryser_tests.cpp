#include "matrix_utils.h"
#include "ryser_algorithms.h"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
using std::vector;
long long calcPermanent(const vector<vector<int>>& matrix, vector<int> cols, int startRow) {
    int n = cols.size();
    if (n == 1) {
        // Base case: If there's only one column left, return its value in the remaining row
        return matrix[startRow][cols[0]];
    }
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        // Create a new column list excluding the current column
        vector<int> nextCols = cols;
        nextCols.erase(nextCols.begin() + i);
        // Recursively calculate the permanent of the submatrix without the current column
        sum += matrix[startRow][cols[i]] * calcPermanent(matrix, nextCols, startRow + 1);
    }
    return sum;
}

// Wrapper function to calculate the permanent of the entire matrix
long long computePermanent(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    vector<int> cols(n);
    for (int i = 0; i < n; ++i) {
        cols[i] = i;
    }
    return calcPermanent(matrix, cols, 0);
}


TEST(PermanentTest, Naive) {
    vector<vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    EXPECT_EQ(computePermanent(matrix), 450); 
}


TEST(PermanentTest, Ryser) {
    int n = 10; 
    auto matrix = generateMatrix(n, 0.3);


    int * matrixFlatten = flattenVector(matrix);
    
    
    long long naivePermanent = computePermanent(matrix);
    long long ryserPermanent = computePermanentRyser(matrixFlatten, n);

    delete[] matrixFlatten;
    
    EXPECT_EQ(naivePermanent, ryserPermanent);
}

TEST(PermanentTest, RyserGreyCode) {
    int n = 10; 
    auto matrix = generateMatrix(n, 0.3);

    int * matrixFlatten = flattenVector(matrix);

    long long naivePermanent = computePermanent(matrix);
    long long ryserGreyCodePermanent = computePermanentRyserGreyCode(matrixFlatten, n);

    delete[] matrixFlatten;

    EXPECT_EQ(naivePermanent, ryserGreyCodePermanent);
}

TEST(PermanentTest, RyserGreyCodeSparse) {
    int n = 10; 
    auto matrix = generateMatrix(n, 0.03);

    int * matrixFlatten = flattenVector(matrix);

    auto sparse = convertToNonZeroElements(matrixFlatten,n);

    long long naivePermanent = computePermanent(matrix);
    long long ryserGreyCodeSparsePermanent = computePermanentRyserGreyCodeSparse(sparse, n);

    delete[] matrixFlatten;

    EXPECT_EQ(naivePermanent, ryserGreyCodeSparsePermanent);
}
