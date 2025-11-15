#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#define CHECK_HIP_ERROR(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " : " \
                  << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_ROCSOLVER_STATUS(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocSOLVER error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


















int main() {
    // Matrix dimensions (m rows, n columns)
    int m = 4;
    int n = 4;
    int lda = m; // Leading dimension of A (must be >= m)
    int min_mn = std::min(m, n);

    // Host matrix A (column-major format required for rocSOLVER)
    // A = { 12, -51,   4,
    //        6,  167, -68,
    //       -4,   24, -41,
    //        0,    0,   0 } (conceptual row-major, but stored column-major below)
    std::vector<double> hA = { 
        12.0,  6.0, -4.0,  1.0,  // Column 0
       -51.0, 167.0, 24.0,  3.0,  // Column 1
       4.0, -68.0, -41.0, 5.0,   // Column 2
         4.02, -68.1, -41.0, 5.3   // Column 3
    };
    
    // Allocate device memory
    double *dA, *dTau;
    CHECK_HIP_ERROR(hipMalloc(&dA, m * n * sizeof(double)));
    // Tau vector stores Householder scalars, size is min(m, n)
    CHECK_HIP_ERROR(hipMalloc(&dTau, min_mn * sizeof(double)));

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), m * n * sizeof(double), hipMemcpyHostToDevice));

    // Create rocSOLVER handle
    rocblas_handle handle;
    CHECK_ROCSOLVER_STATUS(rocblas_create_handle(&handle));

    // Call rocSOLVER DGEQRF function
    // Computes A = Q*R
    // A is overwritten with R (upper triangle) and Householder vectors (below diagonal)
    // Tau contains the Householder scalars
    CHECK_ROCSOLVER_STATUS(rocsolver_dgeqrf(handle, m, n, dA, lda, dTau));

    

    // Copy result (A and Tau) back to host
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dA, m * n * sizeof(double), hipMemcpyDeviceToHost));
    std::vector<double> hTau(min_mn);
    CHECK_HIP_ERROR(hipMemcpy(hTau.data(), dTau, min_mn * sizeof(double), hipMemcpyDeviceToHost));

    // Print the results (R matrix is upper triangular part of hA)
    std::cout << "QR Factorization complete using rocSOLVER." << std::endl;
    std::cout << "R matrix (upper triangular part of A on exit):" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                std::cout << hA[i + j * lda] << "\t"; // Column-major access
            } else {
                std::cout << "0.0\t";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nTau vector (Householder scalars):" << std::endl;
    for (int i = 0; i < min_mn; ++i) {
        std::cout << hTau[i] << std::endl;
    }

    // 2. Generate the Q matrix explicitly (dorgqr)
    // The dA matrix currently holds the implicit Q factors. 
    // We call dorgqr to generate the explicit Q matrix in the *same* memory space dA.
    // The dimensions passed to dorgqr are usually m x m for a full Q,
    // but here we generate an m x n Q to match the original A's dimensions (reduced QR).
    CHECK_ROCSOLVER_STATUS(rocsolver_dorgqr(handle, m, n, min_mn , dA, lda, dTau));


    // Copy result (A and Tau) back to host
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dA, m * n * sizeof(double), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hTau.data(), dTau, min_mn * sizeof(double), hipMemcpyDeviceToHost));

    // Print the results (R matrix is upper triangular part of hA)
    std::cout << "QR Factorization complete using rocSOLVER." << std::endl;
    std::cout << "R matrix (upper triangular part of A on exit):" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                std::cout << hA[i + j * lda] << "\t"; // Column-major access
            } else {
                std::cout << "0.0\t";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nTau vector (Householder scalars):" << std::endl;
    for (int i = 0; i < min_mn; ++i) {
        std::cout << hTau[i] << std::endl;
    }

    
    // Destroy handle and free device memory
    CHECK_ROCSOLVER_STATUS(rocblas_destroy_handle(handle));
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dTau));


    
    return 0;
}
