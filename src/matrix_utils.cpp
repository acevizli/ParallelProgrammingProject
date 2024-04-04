#include "matrix_utils.h"
#include <random>
std::vector<NonZeroElement> convertToNonZeroElements(int* A, int n) {
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

std::vector<std::vector<int>> generateMatrix(int n,double density) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    int numberOfNonZeros = static_cast<int>(n * n * density);

    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<int> distr(0, n);

    for (int i = 0; i < numberOfNonZeros; ++i) {
        int row, col;
        do {
            row = distr(eng) % n; // Random row index
            col = distr(eng) % n; // Random column index
        } while (matrix[row][col] != 0); // Ensure the slot is initially zero to avoid placing two non-zeros in the same spot

        matrix[row][col] = distr(eng); // Assign a random non-zero value (1 to 9) to this position
    }

    return matrix;
}

int * generateMatrixFlatten(int n,double density) {
     int* matrix = new int[n * n]; // Dynamically allocate memory for n*n integers
    
       int numberOfNonZeros = static_cast<int>(n * n * density); // Total non-zero elements based on density

    std::srand(std::time(nullptr)); // Seed for random number generation

    for (int i = 0; i < numberOfNonZeros; ++i) {
        int row, col;
        do {
            row = std::rand() % n; // Random row index
            col = std::rand() % n; // Random column index
        } while (matrix[n * row + col] != 0); // Ensure the slot is initially zero to avoid placing two non-zeros in the same spot

        matrix[row * n + col] = std::rand() % 9 + 1; // Assign a random non-zero value (1 to 9) to this position
    }
    
    return matrix; 
}

int* flattenVector(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return nullptr; // Check for empty input
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    int* flatArray = new int[rows * cols]; // Dynamically allocate memory for the flattened array
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatArray[i * cols + j] = matrix[i][j]; // Flatten the matrix
        }
    }
    
    return flatArray; // Return the pointer to the flattened array
}