
#include <complex.h> 

#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>


#include <stdio.h>
#include <stdlib.h>
//#include <hip/hip_runtime.h>
//#include <rocblas.h>
#include <math.h> // For abs() used in verification

// Helper macro for checking HIP errors
#define CHECK_HIP(func) { \
    hipError_t e = func; \
    if (e != hipSuccess) { \
        printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Helper macro for checking rocBLAS errors
#define CHECK_ROCBLAS(func) { \
    rocblas_status s = func; \
    if (s != rocblas_status_success) { \
        printf("rocBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}



int main() {
    // 1. Setup rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // 2. Define Matrix Dimensions and Batch Count
    rocblas_int M = 2; // Rows of A and C
    rocblas_int N = 2; // Columns of B and C
    rocblas_int K = 2; // Columns of A and Rows of B
    rocblas_int batchCount = 3; // Number of matrix multiplications to perform

    // Define Leading Dimensions (LDA, LDB, LDC) - assuming column-major storage
    rocblas_int ldA = M; 
    rocblas_int ldB = K; 
    rocblas_int ldC = M; 

    // Calculate matrix sizes in elements
    size_t matrix_elements_A = M * K;
    size_t matrix_elements_B = K * N;
    size_t matrix_elements_C = M * N;

    // Calculate matrix sizes in bytes
    size_t matrix_size_A = matrix_elements_A * sizeof(rocblas_double_complex);
    size_t matrix_size_B = matrix_elements_B * sizeof(rocblas_double_complex);
    size_t matrix_size_C = matrix_elements_C * sizeof(rocblas_double_complex);

    // 3. Host Memory Allocation (for initialization, B/C data storage, and verification)

    // A is unique, so we only need to initialize one copy on the host
    rocblas_double_complex *h_A_data = (rocblas_double_complex*)malloc(matrix_size_A);
    // B and C are batched, so allocate space for all of them contiguously on the host
    rocblas_double_complex *h_B_data = (rocblas_double_complex*)malloc(batchCount * matrix_size_B);
    rocblas_double_complex *h_C_data = (rocblas_double_complex*)malloc(batchCount * matrix_size_C);
    rocblas_double_complex *h_C_ref  = (rocblas_double_complex*)malloc(batchCount * matrix_size_C);

    // 4. Initialization of Host Data
    
    // Initialize the single A matrix
    for (int i = 0; i < matrix_elements_A; ++i) {
        h_A_data[i] = rocblas_double_complex{(double)(i + 1), (double)(i + 1)}; 
    }
    
    // Initialize the batched B and C matrices (different data for each batch 'p')
    for (int p = 0; p < batchCount; ++p) {
        for (int i = 0; i < matrix_elements_B; ++i) {
            // B data changes per batch
	  h_B_data[p * matrix_elements_B + i] = rocblas_double_complex{(double)(i + p * 5 + 1), (double)(i + p * 5 + 1)};
        }
        for (int i = 0; i < matrix_elements_C; ++i) {
            h_C_data[p * matrix_elements_C + i] = rocblas_double_complex{0.0, 0.0};
            h_C_ref[p * matrix_elements_C + i] = rocblas_double_complex{0.0, 0.0};
        }
    }

    // 5. Device Memory Allocation (Data Buffers)

    // Allocate memory on the GPU for the single A, and batched B/C
    rocblas_double_complex *d_A_single, *d_B_base, *d_C_base;
    CHECK_HIP(hipMalloc((void**)&d_A_single, matrix_size_A));
    CHECK_HIP(hipMalloc((void**)&d_B_base, batchCount * matrix_size_B));
    CHECK_HIP(hipMalloc((void**)&d_C_base, batchCount * matrix_size_C));

    // 6. Copy Host Data to Device Buffers
    CHECK_HIP(hipMemcpy(d_A_single, h_A_data, matrix_size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_base, h_B_data, batchCount * matrix_size_B, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C_base, h_C_data, batchCount * matrix_size_C, hipMemcpyHostToDevice));

    // 7. Setup Pointer Arrays (This is where the magic happens)

    // Allocate host memory for arrays of pointers
    rocblas_double_complex **h_A_ptrs = (rocblas_double_complex**)malloc(batchCount * sizeof(rocblas_double_complex*));
    rocblas_double_complex **h_B_ptrs = (rocblas_double_complex**)malloc(batchCount * sizeof(rocblas_double_complex*));
    rocblas_double_complex **h_C_ptrs = (rocblas_double_complex**)malloc(batchCount * sizeof(rocblas_double_complex*));

    // Populate the host pointer arrays
    for (int i = 0; i < batchCount; ++i) {
        // !!! KEY STEP: Every pointer in A array points to the single d_A_single buffer !!!
        h_A_ptrs[i] = d_A_single; 
        
        // B and C pointers point to the start of their respective matrix locations within the contiguous blocks
        h_B_ptrs[i] = d_B_base + i * matrix_elements_B;
        h_C_ptrs[i] = d_C_base + i * matrix_elements_C;
    }

    // 8. Allocate Device Memory for the Pointer Arrays (The GPU needs its own copy of the pointers)
    rocblas_double_complex **d_A_ptrs, **d_B_ptrs, **d_C_ptrs;
    CHECK_HIP(hipMalloc((void**)&d_A_ptrs, batchCount * sizeof(rocblas_double_complex*)));
    CHECK_HIP(hipMalloc((void**)&d_B_ptrs, batchCount * sizeof(rocblas_double_complex*)));
    CHECK_HIP(hipMalloc((void**)&d_C_ptrs, batchCount * sizeof(rocblas_double_complex*)));

    // 9. Copy Host Pointer Arrays to Device
    CHECK_HIP(hipMemcpy(d_A_ptrs, h_A_ptrs, batchCount * sizeof(rocblas_double_complex*), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_ptrs, h_B_ptrs, batchCount * sizeof(rocblas_double_complex*), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C_ptrs, h_C_ptrs, batchCount * sizeof(rocblas_double_complex*), hipMemcpyHostToDevice));

    // 10. Define Scalar Alpha and Beta (on the host, passed by pointer)
    rocblas_double_complex alpha = rocblas_double_complex{1.0, 0.0};
    rocblas_double_complex beta = rocblas_double_complex{0.0, 0.0};

    // 11. Execute the rocBLAS ZGEMM Batched Operation
    printf("Executing rocblas_zgemm_batched with %d batches...\n", batchCount);
    CHECK_ROCBLAS(rocblas_zgemm_batched(
        handle, 
        rocblas_operation_none, rocblas_operation_none, // Transpose options (None means no transpose)
        M, N, K,
        &alpha,
        (const rocblas_double_complex* const*)d_A_ptrs, ldA, // Pointer to array of const A pointers
        (const rocblas_double_complex* const*)d_B_ptrs, ldB, // Pointer to array of const B pointers
        &beta,
        d_C_ptrs, ldC, // Pointer to array of C pointers
        batchCount));
    printf("Kernel finished.\n");

    // 12. Copy Result Data Back to Host Contiguous Memory
    CHECK_HIP(hipMemcpy(h_C_data, d_C_base, batchCount * matrix_size_C, hipMemcpyDeviceToHost));

    // 13. Verification (CPU reference using accessor functions)
    for (int p = 0; p < batchCount; ++p) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                rocblas_double_complex sum = rocblas_double_complex{0.0, 0.0};
                for (int k = 0; k < K; ++k) {
                    // Note: h_A_data accesses the single A matrix for all batches now
                    rocblas_double_complex a = h_A_data[m + k * ldA]; 
                    rocblas_double_complex b = h_B_data[p * matrix_elements_B + k + n * ldB];
                    rocblas_double_complex prod = a*b;
		    sum = sum +prod;
                }
                h_C_ref[p * matrix_elements_C + m + n * ldC] = sum;
            }
        }
    }

    // 14. Check Results
    int errors = 0;
    for (int p = 0; p < batchCount; ++p) {
        for (int i = 0; i < matrix_elements_C; ++i) {
	  rocblas_double_complex h_C   = h_C_data[p * matrix_elements_C + i];
           rocblas_double_complex h_C_r   = h_C_ref [p * matrix_elements_C + i];
	    
	  if (h_C != h_C_r) {
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("Verification successful!\n");
    } else {
        printf("Verification failed with %d errors! \n", errors);
    }

    // 15. Cleanup
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));
    CHECK_HIP(hipFree(d_A_single)); // Free single A allocation
    CHECK_HIP(hipFree(d_B_base));
    CHECK_HIP(hipFree(d_C_base));
    CHECK_HIP(hipFree(d_A_ptrs));
    CHECK_HIP(hipFree(d_B_ptrs));
    CHECK_HIP(hipFree(d_C_ptrs));
    free(h_A_data); free(h_B_data); free(h_C_data); free(h_C_ref);
    free(h_A_ptrs); free(h_B_ptrs); free(h_C_ptrs);

    return 0;
}
