#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <ctime>
#include <iomanip>
#include <numeric> // For std::accumulate

using namespace std;

// Define a type for complex numbers and a matrix
using Complex = complex<double>;;
using Matrix = vector<vector<Complex>>;

/**
 * Generates a random complex number.
 */
Complex random_complex() {
    // Use a static random number generator for efficiency
    static mt19937 rng(time(nullptr));
    static uniform_real_distribution<double> dist(0.0, 5.0);
    return Complex(dist(rng), dist(rng));
}

/**
 * Generates a complex, Hermitian, and strongly diagonally dominant matrix.
 * 
 * @param n The dimension of the square matrix (n x n).
 * @param dominance_factor A factor (>1) to ensure strong diagonal dominance.
 */
Matrix generate_hermitian_diagonally_dominant_matrix(int n, double dominance_factor = 1.5) {
    Matrix A(n, vector<Complex>(n));

    // 1. Generate a random complex matrix (off-diagonal part)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Complex value = random_complex();
            A[i][j] = value;
            A[j][i] = conj(value); // Force Hermitian property
        }
    }

    // 2. Calculate row sums of absolute values of off-diagonal elements
    vector<double> row_sums_abs_off_diag(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double current_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                current_sum += abs(A[i][j]);
            }
        }
        row_sums_abs_off_diag[i] = current_sum;
    }

    // 3. Set diagonal elements to be real and strongly dominant
    for (int i = 0; i < n; ++i) {
        // Ensure diagonal element magnitude > dominance_factor * (sum of off-diagonals)
        // Diagonal elements of a Hermitian matrix must be real.
        double new_diag_value = row_sums_abs_off_diag[i] * dominance_factor + 1.0; 
        A[i][i] = Complex(new_diag_value, 0.0);
    }

    return A;
}

/**
 * Prints the matrix.
 */
void print_matrix(const Matrix& A) {
    for (const auto& row : A) {
        for (const auto& val : row) {
            cout << fixed << setprecision(2) << setw(15) << val;
        }
        cout << endl << endl;
    }
}

/**
 * Verifies if the matrix is strongly diagonally dominant (optional).
 */
bool is_hermitian_diagonally_dominant(const Matrix& A) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        double row_sum_abs_off_diag = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                row_sum_abs_off_diag += abs(A[i][j]);
            }
        }
        if (abs(A[i][i]) <= row_sum_abs_off_diag) {
            return false; // Not strictly dominant
        }
    }
    // A full check would also verify the Hermitian property (A[i][j] == conj(A[j][i]))
    return true;
}

// Main function
int main() {
    const int N = 4;
    Matrix hermitian_matrix = generate_hermitian_diagonally_dominant_matrix(N, 1.5);

    cout << "Generated Hermitian and Strongly Diagonally Dominant Matrix:" << endl << endl;
    print_matrix(hermitian_matrix);

    cout << "Verification:" << endl;
    if (is_hermitian_diagonally_dominant(hermitian_matrix)) {
        cout << "The matrix is strongly diagonally dominant." << endl;
    } else {
        cout << "The matrix is NOT strongly diagonally dominant." << endl;
    }

    return 0;
}
