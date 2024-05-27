#include <iostream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <fstream>

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
  int n;
  double* matrix;
 
  if(argc == 3) {
    n = stoi(argv[1]);
    float nonZeroPercentage = atof(argv[2]);

    matrix = new double[n*n];
    memset(matrix, 0, sizeof(double) * n * n);
    srand(time(NULL));
    // initialize randomly
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	double random_value = rand() / (RAND_MAX + 0.0f);
	if(j % 2 == 0) {
	  matrix[i*n+j] = (random_value < nonZeroPercentage) ? (1 + random_value/5) : 0;
	} else {
	  matrix[i*n+j] = (random_value < nonZeroPercentage) ? (1 - random_value/5) : 0;
	}	
      } 
    }
    PrintMatrix(matrix, n);    
  } else if(argc == 2) {
    std::string filename = argv[1];  // Replace with your file name
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return 1;
    }

    int nonzeros;
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
  }

  if(argc > 1) {
    cout << big_perman(matrix, n) << endl;
    delete[] matrix;
  }
  return 0;
}




