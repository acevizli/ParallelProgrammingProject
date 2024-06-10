
#include "matrix_utils.h"
#include "ryser_algorithms.h"
#include "ryser_cuda.h"

#include <cstring>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdexcept> 
using namespace std;
void PrintMatrix(double* matrix, int size){
  int count = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if(matrix[i * size + j] != 0) {
	count++;
      }
    }
  }

  cout << size << " " << count << endl;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if(matrix[i * size + j] != 0) {
        cout << i << " " << j << " " << matrix[i * size + j] << endl;
      }
    }
  }
}


double big_perman (double *a, int m) {
  double x[64];// temporary vector as used by Nijenhuis and Wilf
  double rs;   // row sum of matrix
  double s;    // +1 or -1
  double prod; // product of the elements in vector 'x'
  double p=1.0;  // many results accumulate here, MAY need extra precision
  double *xptr, *aptr; 
  int j, k;
  unsigned long long int i, tn11 = (1ULL<<(m-1))-1ULL;  // tn11 = 2^(n-1)-1
  unsigned long long int gray, prevgray=0, two_to_k;
  
  for (j=0; j<m; j++) {
    rs = 0.0;
    for (k=0; k<m; k++)
      rs += a[j + k*m];  // sum of row j
    x[j] = a[j + (m-1)*m] - rs/2;  // see Nijenhuis and Wilf
    p *= x[j];   // product of the elements in vector 'x'
  }

  for (i=1; i<=tn11; i++) {
    gray=i^(i>>1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    
    two_to_k=1;    // two_to_k = 2 raised to the k power (2^k)
    k=0;
    while (two_to_k < (gray^prevgray))
      {
	two_to_k<<=1;  // two_to_k is a bitmask to find location of 1
	k++;
      }
    s = (two_to_k & gray) ? +1.0 : -1.0;
    prevgray = gray;        
    
    prod = 1.0;
    xptr = (double *)x;
    aptr = &a[k*m];
    for (j=0; j<m; j++)
      {
	*xptr += s * *aptr++;  // see Nijenhuis and Wilf
	prod *= *xptr++;  // product of the elements in vector 'x'
      }
    p += ((i&1ULL)? -1.0:1.0) * prod; 
  }
  
  return((4*(m&1)-2) * p);
}

int main(int argc, char *argv[]){

  if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_name> <no_threads/no_GPUs>" << std::endl;
        return 1;
  }

  int n;
  double* matrix;
  int nonzeros =0;
    std::string filename = argv[1]; 
    std::ifstream file(filename);

    if (filename.empty()) {
        std::cerr << "Error: Filename cannot be empty" << std::endl;
        return 1;
    }

    int thread_count;
    try {
        thread_count = std::stoi(argv[2]);
    } catch (const std::invalid_argument&) {
        std::cerr << "Error: no_threads/no_GPUs must be an integer" << std::endl;
        return 1;
    } catch (const std::out_of_range&) {
        std::cerr << "Error: no_threads/no_GPUs argument is out of range" << std::endl;
        return 1;
    }

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return 1;
    }

    file >> n >> nonzeros;

    matrix = new double[n*n];
    memset(matrix, 0, sizeof(double) * n * n);

    for (int i = 0; i < nonzeros; ++i) {
        int row_id, col_id;
	      double nnz_value;
        file >> row_id >> col_id >> nnz_value;
      	matrix[(row_id * n) + col_id] = nnz_value;
    }
    file.close();  
  
    int* crs_ptrs = (int*)malloc((n + 1) * sizeof(int));
    int* crs_colids = (int*)malloc(nonzeros * sizeof(int));
    double* crs_values = (double*)malloc(nonzeros * sizeof(double));

    // CCS
    int* ccs_ptrs = (int*)malloc((n + 1) * sizeof(int));
    int* ccs_rowids = (int*)malloc(nonzeros * sizeof(int));
    double* ccs_values = (double*)malloc(nonzeros* sizeof(double));
    
    
    convertToCRS(matrix, n, crs_ptrs, crs_colids, crs_values);
    convertToCCS(matrix, n, ccs_ptrs, ccs_rowids, ccs_values);

#if defined(GPU_TEST)
    if(thread_count > 1) {
      double start= omp_get_wtime();
      double result = computePermanentSpaRyserMainMultiGPU(thread_count, n, nonzeros, crs_ptrs, crs_colids, crs_values, ccs_ptrs, ccs_rowids, ccs_values);
      double end = omp_get_wtime();
      auto time = end - start;
      cout << std::scientific << std::setprecision(2) << result <<" "<<std::defaultfloat <<std::setprecision(6) << time << std::endl;
    }
    else {
      double start= omp_get_wtime();
      double result = computePermanentSpaRyserMain(n, nonzeros, crs_ptrs, crs_colids, crs_values, ccs_ptrs, ccs_rowids, ccs_values);
      double end = omp_get_wtime();
      auto time = end - start;
      cout << std::scientific << std::setprecision(2) << result << " " <<std::defaultfloat <<std::setprecision(6) << time << std::endl;
    }
#elif defined(CPU_PAR_TEST)
    omp_set_num_threads(thread_count);
      #pragma omp parallel
    {
        int threads = omp_get_num_threads();        
        #pragma omp single
        {
          int threads = omp_get_num_threads();
          if(threads != thread_count) {
              std::cerr << "WARNING: OMP set threads failed Performance will be impacted, Requested thread count: "
              << thread_count <<" Current Thread count: "<<threads<< std::endl;
          }
        }
    }
    double start_openmp= omp_get_wtime();
    double result = computePermanentSpaRyserPar(n,crs_ptrs,crs_colids,crs_values,ccs_ptrs,ccs_rowids,ccs_values);
    double end_openmp = omp_get_wtime();
    auto time = end_openmp - start_openmp;
    cout << std::scientific << std::setprecision(2) << result << " " << std::defaultfloat <<std::setprecision(6) << time <<std::endl;
  #endif
    delete[] matrix;
  return 0;
}



