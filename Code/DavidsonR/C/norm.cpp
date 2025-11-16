#include <iostream>
#include <vector>
#include <cmath>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP_ERROR(call)                                                  \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__        \
                      << " : " << hipGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_ROCBLAS_STATUS(call)                                             \
    do {                                                                       \
        rocblas_status status = call;                                          \
        if (status != rocblas_status_success) {                                \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " : " << status << std::endl;                         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    // Matrix dimensions: M rows, N columns (each column is a vector)
    rocblas_int M = 5;
    rocblas_int N = 3;
    rocblas_int incx = 1; // Stride within each vector (contiguous elements)
    rocblas_stride stride_x = M * incx; // Stride between the start of consecutive vectors
    rocblas_int batch_count = N; // Number of vectors to process
    
    // --- Host Data ---
    std::vector<double> h_A(M * N);
    std::vector<double> h_norms(N);
    std::vector<double> h_norms_ref(N, 0.0);

    // Initialize host matrix A (column-major)
    // Matrix A:
    // 1.0  6.0  11.0
    // 2.0  7.0  12.0
    // 3.0  8.0  13.0
    // 4.0  9.0  14.0
    // 5.0  10.0 15.0
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            h_A[j * M + i] = (double)(i + 1 + j * M);
        }
    }

    // Calculate reference norms on CPU
    for (int j = 0; j < N; ++j) {
        double sum_sq = 0.0;
        for (int i = 0; i < M; ++i) {
            sum_sq += h_A[j * M + i] * h_A[j * M + i];
        }
        h_norms_ref[j] = std::sqrt(sum_sq);
    }

    // --- Device Data ---
    double *d_A, *d_norms;
    CHECK_HIP_ERROR(hipMalloc(&d_A, M * N * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&d_norms, N * sizeof(double)));

    // Copy host matrix to device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), M * N * sizeof(double), hipMemcpyHostToDevice));

    // --- rocBLAS Initialization ---
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // Set pointer mode to device, as results are stored on the device
    CHECK_ROCBLAS_STATUS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    // --- rocBLAS Call (Strided Batched NRM2) ---
    std::cout << "Calling rocblas_dnrm2_strided_batched..." << std::endl;
    CHECK_ROCBLAS_STATUS(rocblas_dnrm2_strided_batched(
        handle,
        M,               // n: length of each vector
        d_A,             // x: device pointer to the matrix/vectors
        incx,            // incx: stride within each vector (1 for column-major matrix)
        stride_x,        // stride_x: stride between consecutive vectors in memory
        batch_count,     // batch_count: number of vectors (columns)
        d_norms          // result: device pointer to store norms
    ));

    // --- Copy results back to host ---
    CHECK_HIP_ERROR(hipMemcpy(h_norms.data(), d_norms, N * sizeof(double), hipMemcpyDeviceToHost));

    // --- Verification and Cleanup ---
    std::cout << "Verification:" << std::endl;
    bool success = true;
    for (int i = 0; i < N; ++i) {
        // Use a small tolerance for floating-point comparison
        if (std::abs(h_norms[i] - h_norms_ref[i]) > 1e-9) {
            std::cerr << "Error: Norm " << i << " mismatch! Expected: " 
                      << h_norms_ref[i] << ", Got: " << h_norms[i] << std::endl;
            success = false;
        } else {
            std::cout << "Norm " << i << ": " << h_norms[i] << " (matches reference)" << std::endl;
        }
    }

    if (success) {
        std::cout << "All norms verified successfully!" << std::endl;
    }

    // Cleanup
    CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_norms));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
