
#include <complex.h> 

#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>
//#include <cblas.h>



typedef rocblas_double_complex ZC;


#include <stdio.h>
#include <stdlib.h>
//#include <hip/hip_runtime.h>
//#include <rocblas.h>
#include <math.h> // For abs() used in verification

// Helper macro for checking HIP errors
#define CHECK_HIP(func) {			\
    hipError_t e = func;			\
    if (e != hipSuccess) {						\
      printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      exit(EXIT_FAILURE);						\
    }									\
  }

// Helper macro for checking rocBLAS errors
#define CHECK_ROCBLAS(func) {					\
    rocblas_status s = func;					\
    if (s != rocblas_status_success) {				\
      printf("rocBLAS error at %s:%d\n", __FILE__, __LINE__);	\
      exit(EXIT_FAILURE);					\
    }								\
  }




// Helper function for host-side matrix multiplication verification (standard C++ implementation)
void cpu_zgemm_batched(int M, int N, int K, ZC alpha, 
               ZC* A, rocblas_stride ldA,
               ZC* B, rocblas_stride ldB,
               ZC beta,
	       ZC* C, rocblas_stride ldC,
               int batchCount
	       ) {

  // Calculate matrix sizes in elements
  size_t matrix_elements_A = M * K;
  size_t matrix_elements_B = K * N;
  size_t matrix_elements_C = M * N;


  for (int p = 0; p < batchCount; ++p) {
      for (int m = 0; m < M; ++m) {
	for (int n = 0; n < N; ++n) {
	  ZC sum = ZC{0.0, 0.0};
	  for (int k = 0; k < K; ++k) {
	    // Note: h_A_data accesses the single A matrix for all batches now
	    ZC a = A[m + k * ldA]; 
	    ZC b = B[p * matrix_elements_B + k + n * ldB];
	    sum = sum +a*b;
	  }
	  C[p * matrix_elements_C + m + n * ldC] = sum +  C[p * matrix_elements_C + m + n * ldC]*beta;
	}
      }
  }
}

// Helper function for host-side matrix multiplication verification (standard C++ implementation)
void cpu_zgemm_batched_b(int M, int N, int K, ZC alpha, 
               ZC* A, rocblas_stride ldA,
               ZC* B, rocblas_stride ldB,
               ZC beta,
	       ZC* C, rocblas_stride ldC,
               int batchCount
	       ) {

  // Calculate matrix sizes in elements
  size_t matrix_elements_A = M * K;
  size_t matrix_elements_B = K * N;
  size_t matrix_elements_C = M * N;


  for (int p = 0; p < batchCount; ++p) {
      for (int m = 0; m < M; ++m) {
	for (int n = 0; n < N; ++n) {
	  ZC sum = ZC{0.0, 0.0};
	  for (int k = 0; k < K; ++k) {
	    // Note: h_A_data accesses the single A matrix for all batches now
	    ZC a = A[p * matrix_elements_A +m + k * ldA]; 
	    ZC b = B[k + n * ldB];
	    sum = sum +a*b;
	  }
	  C[p * matrix_elements_C + m + n * ldC] = sum +  C[p * matrix_elements_C + m + n * ldC]*beta;
	}
      }
  }
}






void pre_gpu_gemm(
	 int M, int N, int K, 
	 ZC** A, rocblas_stride ldA, ZC *d_A,
	 ZC** B, rocblas_stride ldB, ZC *d_B,
	 ZC** C, rocblas_stride ldC, ZC *d_C,
	 int batchCount
		  ) {

  // Calculate matrix sizes in elements
  size_t matrix_elements_A = M * K;
  size_t matrix_elements_B = K * N;
  size_t matrix_elements_C = M * N;

  // Populate the host pointer arrays
  for (int i = 0; i < batchCount; ++i) {
    // !!! KEY STEP: Every pointer in A array points to the single d_A_single buffer !!!
    A[i] = d_A; 
    
    // B and C pointers point to the start of their respective matrix locations within the contiguous blocks
    B[i] = d_B + i * matrix_elements_B;
    C[i] = d_C + i * matrix_elements_C;
  }

}



void gpu_zgemm_batched(
	       rocblas_handle handle,
	       int M, int N, int K, ZC alpha, 
               ZC** A, rocblas_stride ldA,
               ZC** B, rocblas_stride ldB,
               ZC beta,
	       ZC** C, rocblas_stride ldC,
               int batchCount
	       ) {

  CHECK_ROCBLAS(
     rocblas_zgemm_batched(
	handle, 
        rocblas_operation_none, rocblas_operation_none, // Transpose options (None means no transpose)
        M, N, K,
        &alpha,
        (const ZC* const*)A, ldA, // Pointer to array of const A pointers
        (const ZC* const*)B, ldB, // Pointer to array of const B pointers
        &beta,
        C, ldC, // Pointer to array of C pointers
        batchCount)
		);
  
  
}



#ifdef MAIN_HERE

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
    size_t matrix_size_A = matrix_elements_A * sizeof(ZC);
    size_t matrix_size_B = matrix_elements_B * sizeof(ZC);
    size_t matrix_size_C = matrix_elements_C * sizeof(ZC);

    // 3. Host Memory Allocation (for initialization, B/C data storage, and verification)

    // A is unique, so we only need to initialize one copy on the host
    ZC *h_A_data = (ZC*)malloc(matrix_size_A);
    // B and C are batched, so allocate space for all of them contiguously on the host
    ZC *h_B_data = (ZC*)malloc(batchCount * matrix_size_B);
    ZC *h_C_data = (ZC*)malloc(batchCount * matrix_size_C);
    ZC *h_C_ref  = (ZC*)malloc(batchCount * matrix_size_C);

    // 4. Initialization of Host Data
    
    // Initialize the single A matrix
    for (int i = 0; i < matrix_elements_A; ++i) {
        h_A_data[i] = ZC{(double)(i + 1), (double)(i + 1)}; 
    }
    
    // Initialize the batched B and C matrices (different data for each batch 'p')
    for (int p = 0; p < batchCount; ++p) {
        for (int i = 0; i < matrix_elements_B; ++i) {
            // B data changes per batch
	  h_B_data[p * matrix_elements_B + i] = ZC{(double)(i + p * 5 + 1), (double)(i + p * 5 + 1)};
        }
        for (int i = 0; i < matrix_elements_C; ++i) {
            h_C_data[p * matrix_elements_C + i] = ZC{0.0, 0.0};
            h_C_ref[p * matrix_elements_C + i] = ZC{0.0, 0.0};
        }
    }

    // 5. Device Memory Allocation (Data Buffers)

    // Allocate memory on the GPU for the single A, and batched B/C
    ZC *d_A_single, *d_B_base, *d_C_base;
    CHECK_HIP(hipMalloc((void**)&d_A_single, matrix_size_A));
    CHECK_HIP(hipMalloc((void**)&d_B_base, batchCount * matrix_size_B));
    CHECK_HIP(hipMalloc((void**)&d_C_base, batchCount * matrix_size_C));

    // 6. Copy Host Data to Device Buffers
    CHECK_HIP(hipMemcpy(d_A_single, h_A_data, matrix_size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_base, h_B_data, batchCount * matrix_size_B, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C_base, h_C_data, batchCount * matrix_size_C, hipMemcpyHostToDevice));

    // 7. Setup Pointer Arrays (This is where the magic happens)

    // Allocate host memory for arrays of pointers
    ZC **h_A_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));
    ZC **h_B_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));
    ZC **h_C_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));

    // Populate the host pointer arrays
    pre_gpu_gemm(M,N,K,h_A_ptrs,ldA,d_A_single,
		      h_B_ptrs,ldB,d_B_base,
		      h_C_ptrs,ldC,d_C_base,
		      batchCount);
    
    // 8. Allocate Device Memory for the Pointer Arrays (The GPU needs its own copy of the pointers)
    ZC **d_A_ptrs, **d_B_ptrs, **d_C_ptrs;
    CHECK_HIP(hipMalloc((void**)&d_A_ptrs, batchCount * sizeof(ZC*)));
    CHECK_HIP(hipMalloc((void**)&d_B_ptrs, batchCount * sizeof(ZC*)));
    CHECK_HIP(hipMalloc((void**)&d_C_ptrs, batchCount * sizeof(ZC*)));

    // 9. Copy Host Pointer Arrays to Device
    CHECK_HIP(hipMemcpy(d_A_ptrs, h_A_ptrs, batchCount * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_ptrs, h_B_ptrs, batchCount * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C_ptrs, h_C_ptrs, batchCount * sizeof(ZC*), hipMemcpyHostToDevice));

    // 10. Define Scalar Alpha and Beta (on the host, passed by pointer)
    ZC alpha = ZC{1.0, 0.0};
    ZC beta = ZC{0.0, 0.0};

    // 11. Execute the rocBLAS ZGEMM Batched Operation
    printf("Executing rocblas_zgemm_batched with %d batches...\n", batchCount);

    gpu_zgemm_batched(handle,M,N,K, alpha, d_A_ptrs,ldA,d_B_ptrs,ldB,beta, d_C_ptrs,ldC,batchCount);
    printf("Kernel finished.\n");

    // 12. Copy Result Data Back to Host Contiguous Memory
    CHECK_HIP(hipMemcpy(h_C_data, d_C_base, batchCount * matrix_size_C, hipMemcpyDeviceToHost));

    // 13. Verification (CPU reference using accessor functions)
    cpu_zgemm_batched(M,N,K, alpha, h_A_data,ldA,h_B_data,ldB,beta, h_C_ref,ldC,batchCount);


/*
    for (int p = 0; p < batchCount; ++p) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                ZC sum = ZC{0.0, 0.0};
                for (int k = 0; k < K; ++k) {
                    // Note: h_A_data accesses the single A matrix for all batches now
                    ZC a = h_A_data[m + k * ldA]; 
                    ZC b = h_B_data[p * matrix_elements_B + k + n * ldB];
                    ZC prod = a*b;
		    sum = sum +prod;
                }
                h_C_ref[p * matrix_elements_C + m + n * ldC] = sum;
            }
        }
    }
*/
    // 14. Check Results
    int errors = 0;
    for (int p = 0; p < batchCount; ++p) {
        for (int i = 0; i < matrix_elements_C; ++i) {
	  ZC h_C   = h_C_data[p * matrix_elements_C + i];
           ZC h_C_r   = h_C_ref [p * matrix_elements_C + i];
	    
	  if (h_C != h_C_r) {
	    std::cout << "(" << h_C << "vs" << h_C_r << ")"; 
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
#endif
