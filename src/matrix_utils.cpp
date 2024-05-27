#include "matrix_utils.h"
#include <cstring>
#include <random>
std::vector<NonZeroElement> convertToNonZeroElements(double* A, int n) {
    std::vector<NonZeroElement> nonZeroElements;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i * n + j] != 0) {
                nonZeroElements.push_back({i, j, A[i * n + j]});
            }
        }
    }
    return nonZeroElements;
}

std::vector<std::vector<double>> generateMatrix(int n,double density) {
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0));
    int numberOfNonZeros = static_cast<int>(n * n * density);

    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<int> distr(0, n - 1);
    std::uniform_real_distribution<double> distr2(0.0, 1.0);

    for (int i = 0; i < numberOfNonZeros; ++i) {
        int row, col;
        do {
            row = distr(eng); // Random row index
            col = distr(eng); // Random column index
        } while (matrix[row][col] != 0); // Ensure the slot is initially zero to avoid placing two non-zeros in the same spot

        matrix[row][col] = distr2(eng); // Assign a random non-zero value (1 to 9) to this position
    }

    return matrix;
}

double * generateMatrixFlatten(int n,double density) {
    double* matrix = new double[n * n]; // Dynamically allocate memory for n*n integers
    memset(matrix,0,sizeof(int) * n * n);
    int numberOfNonZeros = static_cast<int>(n * n * density); // Total non-zero elements based on density

    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<int> distr(0, n - 1);
    std::uniform_real_distribution<double> distr2(0.0, 1.0);

    for (int i = 0; i < numberOfNonZeros; ++i) {
        int row, col;
        do {
            row = distr(eng); // Random row index
            col = distr(eng); // Random column index
        } while (matrix[n * row + col] != 0); // Ensure the slot is initially zero to avoid placing two non-zeros in the same spot

        matrix[row * n + col] = distr2(eng); // Assign a random non-zero value (1 to 9) to this position
    }
    
    return matrix; 
}

double* flattenVector(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return nullptr; // Check for empty input
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    double* flatArray = new double[rows * cols]; // Dynamically allocate memory for the flattened array
    memset(flatArray,0,rows * cols * sizeof(int));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatArray[i * cols + j] = matrix[i][j]; // Flatten the matrix
        }
    }
    
    return flatArray; // Return the pointer to the flattened array
}


void convertToCRS(double* A, int n, int* crs_ptrs, int* crs_colids, double* crs_values) {

    //fill the crs_ptrs array
    memset(crs_ptrs, 0, (n + 1) * sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(A[i * n + j] != 0){
                crs_ptrs[i+1]++;
            }
        }
    }

    //now we have cumulative ordering of crs_ptrs.
    for(int i = 1; i <= n; i++) {
        crs_ptrs[i] += crs_ptrs[i-1];
    }

    // we set crs_colids such that for each element, it holds the related column of that element
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(A[i * n + j] != 0){
                int index = crs_ptrs[i];

                crs_colids[index] = j;
                crs_values[index] = A[i * n + j];

                crs_ptrs[i] = crs_ptrs[i] + 1; 
            }
        }
    }

    for(int i = n; i > 0; i--) {
        crs_ptrs[i] = crs_ptrs[i-1];
    }
    
    crs_ptrs[0] = 0;

}


void convertToCCS(double* A, int n, int* ccs_ptrs, int* ccs_rowids, double* ccs_values) {

    //fill the crs_ptrs array
    memset(ccs_ptrs, 0, (n + 1) * sizeof(int));
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            if(A[i * n + j] != 0){
                ccs_ptrs[j+1]++;
            }
        }
    }

    //now we have cumulative ordering of crs_ptrs.
    for(int i = 1; i <= n; i++) {
        ccs_ptrs[i] += ccs_ptrs[i-1];
    }

    // we set crs_colids such that for each element, it holds the related column of that element
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            if(A[i * n + j] != 0){
                int index = ccs_ptrs[j];

                ccs_rowids[index] = i;
                ccs_values[index] = A[i * n + j];

                ccs_ptrs[j] = ccs_ptrs[j] + 1; 
            }
        }
    }

    for(int i = n; i > 0; i--) {
        ccs_ptrs[i] = ccs_ptrs[i-1];
    }
    ccs_ptrs[0] = 0;
}
