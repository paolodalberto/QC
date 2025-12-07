#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <ctime>
#include <iomanip>
#include <numeric> // For std::accumulate

#include "davidson.h"

/**
 * Verifies if the matrix is strongly diagonally dominant (optional).
 */
bool is_diagonally_dominant(Matrix& A) {
    int n = A.N;
    for (int i = 0; i < n; ++i) {
        NORM_TYPE row_sum_abs_off_diag = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
	      row_sum_abs_off_diag += std::abs(A.matrix[A.ind(i,j)]);
            }
        }
        if ( std::abs(A.matrix[A.ind(i,i)]) <= row_sum_abs_off_diag) {
            return false; // Not strictly dominant
        }
    }
    // A full check would also verify the Hermitian property (A[i][j] == conj(A[j][i]))
    return true;
}




#if(TYPE_OPERAND==3  ||  TYPE_OPERAND==4) 

/**
 * Generates a random complex number.
 */
ZC random_zc() {
    // Use a static random number generator for efficiency
  static std::mt19937 rng(time(nullptr));
  static std::uniform_real_distribution<double> dist(0.0, 5.0);
  return ZC((NORM_TYPE)dist(rng), (NORM_TYPE)dist(rng));
}



void generate_diagonally_dominant_matrix(Matrix &H, double dominance_factor = 1.5) {

    ZC *A = H.matrix;
    int n = H.N;
    // 1. Generate a random complex matrix (off-diagonal part)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            ZC value = random_zc();
            A[H.ind(i,j)] = value;
            A[H.ind(j,i)] = std::conj(value); // Force Hermitian property
        }
    }

    // 2. Calculate row sums of absolute values of off-diagonal elements
    std::vector<NORM_TYPE> row_sums_abs_off_diag(n, 0.0);
    for (int i = 0; i < n; ++i) {
        NORM_TYPE current_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
	      current_sum += std::abs(A[H.ind(i,j)]);
            }
        }
        row_sums_abs_off_diag[i] = current_sum;
    }

    // 3. Set diagonal elements to be real and strongly dominant
    for (int i = 0; i < n; ++i) {
        // Ensure diagonal element magnitude > dominance_factor * (sum of off-diagonals)
        // Diagonal elements of a Hermitian matrix must be real.
        NORM_TYPE new_diag_value = row_sums_abs_off_diag[i] * dominance_factor + 1.0; 
        A[H.ind(i,i)] = ZC{new_diag_value, 0.0};
    }
}

#else


/**
 * Generates a random complex number.
 */
ZC random_zc() {
    // Use a static random number generator for efficiency
  static std::mt19937 rng(time(nullptr));
  static std::uniform_real_distribution<double> dist(0.0, 5.0);
  return ZC((NORM_TYPE)dist(rng));
}



void generate_diagonally_dominant_matrix(Matrix &H, double dominance_factor = 1.5) {

    ZC *A = H.matrix;
    int n = H.N;
    // 1. Generate a random complex matrix (off-diagonal part)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            ZC value = random_zc();
            A[H.ind(i,j)] = value;
            A[H.ind(j,i)] = value; // Force Hermitian property
        }
    }

    // 2. Calculate row sums of absolute values of off-diagonal elements
    std::vector<NORM_TYPE> row_sums_abs_off_diag(n, 0.0);
    for (int i = 0; i < n; ++i) {
        NORM_TYPE current_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
	      current_sum += std::abs(A[H.ind(i,j)]);
            }
        }
        row_sums_abs_off_diag[i] = current_sum;
    }

    // 3. Set diagonal elements to be real and strongly dominant
    for (int i = 0; i < n; ++i) {
        // Ensure diagonal element magnitude > dominance_factor * (sum of off-diagonals)
        // Diagonal elements of a Hermitian matrix must be real.
        NORM_TYPE new_diag_value = row_sums_abs_off_diag[i] * dominance_factor + 1.0; 
        A[H.ind(i,i)] = ZC{new_diag_value};
    }
}


#endif


#ifdef HERM_MAIN
// Main function
int main() {
    const int N = 4;
    Matrix H{N,N,N,N};
    H.alloc(true,false);
    generate_diagonally_dominant_matrix(H, 1.5);

    std::cout << "Generated Hermitian and Strongly Diagonally Dominant Matrix:" << std::endl << std::endl;
    H.print(true);

    std::cout << "Verification:" << std::endl;
    if (is_diagonally_dominant(H)) {
        std::cout << "The matrix is strongly diagonally dominant." << std::endl;
    } else {
        std::cout << "The matrix is NOT strongly diagonally dominant." << std::endl;
    }

    return 0;
}
#endif
