


/*********************************************************
 * The Idea is simple: The computation of a Gate G on a state S is
 * expressed by a a kronecker computation (x) as 
 *
 * (I_n (x) G (x) I_k) * S
 *
 * I_n stands for an identity matrix and n amd k mean the number of
 * bit that is the Gate applied to m bits the G is applied to bit k,
 * k+1, ... k+m-1 and the total number of bits is n + m + k and the
 * state is of size 2^(n+m+k). 
 *
 * We showed this in the Python implementation and also we show that
 * this boils down to the computation of 2^n matrix multiplicaitons
 * (strided) G over the remainder state S[0... 2^(k+m)] as a matrix of
 * size 2^m x 2^k ... so far brilliant
 *
 **************/


#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>



// we define the computation double complex 

#define TYPE_OPERAND 4 
#include "davidson.h"  // definition of matrices 
#include "circuit.h"   // definition of Gate and Circuit







int main() {
  rocblas_handle handle;
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
    
  // Matrix dimensions: A (N x K), B (K x N), C (N x N)
  const int N = 4;
  const int K = 4;
  const int BATCH_COUNT = 3;
  
  // Define strides (memory distance between start of matrices)
  const rocblas_stride strideA = N * K;
  const rocblas_stride strideB = K * N;
  const rocblas_stride strideC = N * N;
  const size_t total_elements_A = strideA * BATCH_COUNT;
  const size_t total_elements_B = strideB * BATCH_COUNT;
  const size_t total_elements_C = strideC * BATCH_COUNT;
  
  // --- 1. Host side data initialization ---
  std::vector<ZC> h_A(total_elements_A);
    std::vector<ZC> h_B(total_elements_B);
    std::vector<ZC> h_C(total_elements_C, make_complex(0.0, 0.0));
    // A separate CPU ground truth vector for verification
    std::vector<ZC> h_C_cpu_verify(total_elements_C, make_complex(0.0, 0.0));

    // Fill A, B, and C with some values (example: 1.0 + 0.0i, etc.)
    for (size_t i = 0; i < total_elements_A; ++i) h_A[i] = make_complex(1.0, 0.0);
    for (size_t i = 0; i < total_elements_B; ++i) h_B[i] = make_complex(0.5, 0.0);
    // Initialize C with ones so we can test the beta scaling
    for (size_t i = 0; i < total_elements_C; ++i) h_C[i] = make_complex(1.0, 0.0); 

    // Initialize the CPU verification array with the same start values as h_C
    h_C_cpu_verify = h_C;

    // Scaling factors: C = 1.0 * A * B + 1.0 * C 
    ZC alpha = make_complex(1.0, 0.0);
    ZC beta  = make_complex(1.0, 0.0);

    // --- 2. Device memory allocation and data transfer ---
    ZC *d_A, *d_B, *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, total_elements_A * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, total_elements_B * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, total_elements_C * sizeof(ZC)));

    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), total_elements_A * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), total_elements_B * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, h_C.data(), total_elements_C * sizeof(ZC), hipMemcpyHostToDevice));

    // --- 3. Perform the rocBLAS computation ---
    // Here we compute C = alpha * A * B + beta * C
    CHECK_ROCBLAS(rocblas_zgemm_strided_batched(
        handle,
        rocblas_operation_none, 
        rocblas_operation_none, 
        N,           // M
        N,           // N
        K,           // K
        &alpha,      
        d_A,         
        N,           // lda
        strideA,     
        d_B,         
        K,           // ldb
        strideB,     
        &beta,       
        d_C,         // Output pointer
        N,           // ldc
        strideC,     
        BATCH_COUNT  
    ));

    // --- 4. Verification Step ---
    // Synchronize the device and copy results back to host
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::vector<ZC> h_C_gpu_result(total_elements_C);
    CHECK_HIP_ERROR(hipMemcpy(h_C_gpu_result.data(), d_C, total_elements_C * sizeof(ZC), hipMemcpyDeviceToHost));

    // Compute the CPU ground truth result
    cpu_zgemm(N, N, K, alpha, h_A, strideA, h_B, strideB, beta, h_C_cpu_verify, strideC, BATCH_COUNT);

    // Compare results
    bool success = true;
    double tolerance = 1e-9;
    for (size_t i = 0; i < total_elements_C; ++i) {
        double gpu_real = hipCreal(h_C_gpu_result[i]);
        double cpu_real = hipCreal(h_C_cpu_verify[i]);
        double gpu_imag = hipCimag(h_C_gpu_result[i]);
        double cpu_imag = hipCimag(h_C_cpu_verify[i]);

        if (std::abs(gpu_real - cpu_real) > tolerance || std::abs(gpu_imag - cpu_imag) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU (" 
                      << gpu_real << ", " << gpu_imag << ") vs CPU (" 
                      << cpu_real << ", " << cpu_imag << ")" << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "rocBLAS strided batched ZGEMM test PASSED!" << std::endl;
    } else {
        std::cout << "rocBLAS strided batched ZGEMM test FAILED!" << std::endl;
    }

    // --- 5. Cleanup ---
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
 }
