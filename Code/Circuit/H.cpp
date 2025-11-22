


#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>

/*
#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas.h>
#include <cstdlib>
*/
#define CHECK_HIP_ERROR(call) do {                                   \
    hipError_t err = call;                                           \
    if (err != hipSuccess) {                                         \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " : " << hipGetErrorString(err) << std::endl;   \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

#define CHECK_ROCBLAS_ERROR(call) do {                                    \
    rocblas_status status = call;                                         \
    if (status != rocblas_status_success) {                               \
        std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__;  \
        std::cerr << " : " << status << std::endl;                        \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

typedef hipDoubleComplex ZC;


// Helper function to initialize complex numbers
ZC make_complex(double r, double i) {
    return hipCmplx(r, i);
}

// Helper function for host-side matrix multiplication verification (standard C++ implementation)
void cpu_zgemm(int M, int N, int K, ZC alpha, 
               const std::vector<ZC>& A, rocblas_stride strideA,
               const std::vector<ZC>& B, rocblas_stride strideB,
               ZC beta, std::vector<ZC>& C, rocblas_stride strideC,
               int batchCount) {
    for (int batch = 0; batch < batchCount; ++batch) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                // Calculate original C value scaled by beta
                ZC c_val = C[batch * strideC + m + n * M];
                ZC sum = make_complex(0.0, 0.0);

                for (int k_idx = 0; k_idx < K; ++k_idx) {
                    // C++ complex multiplication (hipCmulf) and addition (hipCaddf) are used here
                    ZC a_val = A[batch * strideA + m + k_idx * M];
                    ZC b_val = B[batch * strideB + k_idx + n * K];
                    sum = hipCadd(sum, hipCmul(a_val, b_val));
                }
                
                // C = alpha * (A*B) + beta * C
                ZC alpha_sum = hipCmul(alpha, sum);
                ZC beta_c = hipCmul(beta, c_val);
                C[batch * strideC + m + n * M] = hipCadd(alpha_sum, beta_c);
            }
        }
    }
}




struct matrix {

  // struct as a class members are all public
  int m;                       // rows
  int n;                       // cols
  int M;                       // Maximum rows LD
  int N;                       // Maximum cols
  ZC *matrix = 0; // host  
  ZC *d_matrix =0;            // device

  void free() {
    if (matrix) std::free(matrix);
    if (d_matrix) CHECK_HIP_ERROR(hipFree(d_matrix));
  }
  void alloc(bool host , bool device  ) {
    if (size()>0) {
      //printf(" Allocated %d * %d = %d elements \n", M, N,M*N);
      if (host and matrix==0)     matrix = (double*) std::calloc(M*N,sizeof(ZC));
      if (matrix == NULL) {
	std::cerr << "Memory allocation failed!" << std::endl;
	// Handle the error, e.g., exit, throw an exception, or attempt recovery
	exit(EXIT_FAILURE);
      }
      if (device and matrix_d==0) CHECK_HIP_ERROR(hipMalloc(&d_matrix, M*N* sizeof(ZC)));		      
    }
  }
  void readfromdevice() {
         CHECK_HIP_ERROR(hipMemcpy(matrix , d_matrix, size() * sizeof(ZC), hipMemcpyHostToDevice));
  }
  void writetodevice() {
         CHECK_HIP_ERROR(hipMemcpy(d_matrix , matrix, size() * sizeof(ZC), hipMemcpyHostToDevice));

  }

  void init() {
    for (int i = 0; i < m * n; ++i) matrix[i] = {i % 11, i%17};;
  };
  void zero() {
    for (int i = 0; i < m * n; ++i) matrix[i] = { 0.0, 0.0 };
  };
  int ind(int i, int j, bool t=false)    {
    if (t)
      return i*n +j;
    else
      return i +m*j;
    
  }
  
  int size() { return m*n; } 
  void print(bool t=false) {
    int MM = (m>10)?10: m;
    int NN = (n>10)?10: n;
    ZC z;
    printf("Column Major M,N = %d,%d \n", m,n);
    
    if (t)
	for (int i = 0; i < MM; ++i) {
	  for (int j = 0; j < NN; ++j) {
	    z =  matrix[ind(i,j)];
	    //printf("%0.2f %d %d %d ", matrix[ind(i,j)],i,j,ind(i,j));
	     std::cout << "(" << z.x << ", " << z.y  << ")"; 
	}
	printf("\n");
      }
  };
};
typedef struct matrix Matrix;


struct circuit {
  std::string name;
  int    bit_number;
  Matrix &I; // Input state : 2^n x1 
  Matrix &U; // Gate matrix gxg 
  Matrix &O; // output state
  int m=0;
  int n=0;
  int k=0;
  static ZC alpha = {1.0, 0.0};
  static ZC beta  = {0.0, 0.0};  // beta = 0.0 + 0.0i
  void init() {
    U.alloc(true, true);
    int B = I.m; // this is the state 2^n 
    int batch_count = B - (1>>bit_number+U.m);
    int m = U.m;
    int n = 1>>bit_number;
    int k = U.n;
  }

  void free() {
    U.free();
  }
  
  
  void step(rocblas_handle handle) {

    CHECK_ROCBLAS_ERROR(
        rocblas_zgemm_batched(
	     handle,
	     rocblas_operation_none, rocblas_operation_none, // No transpose for A and B
	     m, n, k,
	     &alpha,
	     (const ZC* const*)U.d_matrix, U.m,
	     (const ZC* const*)I.d_matrix, U.m,
	     &beta,
	     O.d_matrix, U.m,
	     batch_count
			      )
			);
  }
};

typdef struct circuit Circuit;

struct schedule {
  std::vector<std::vector<Circuit>> schedule; 

  void init(){
    for (std::vector<Circuit> &level  : schedule)
      for (Circuit h : level )
	h.init
    
  }
  
  
  void step(rocblas_handle handle) {

    CHECK_ROCBLAS_ERROR(
        rocblas_zgemm_batched(
	     handle,
	     rocblas_operation_none, rocblas_operation_none, // No transpose for A and B
	     m, n, k,
	     &alpha,
	     (const ZC* const*)U.d_matrix, U.m,
	     (const ZC* const*)I.d_matrix, U.m,
	     &beta,
	     O.d_matrix, U.m,
	     batch_count
			      )
			);
  }
};

typdef struct circuit Circuit;





int main() {
      rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

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
    std::vector<hipDoubleComplex> h_A(total_elements_A);
    std::vector<hipDoubleComplex> h_B(total_elements_B);
    std::vector<hipDoubleComplex> h_C(total_elements_C, make_complex(0.0, 0.0));
    // A separate CPU ground truth vector for verification
    std::vector<hipDoubleComplex> h_C_cpu_verify(total_elements_C, make_complex(0.0, 0.0));
`%
    // Fill A, B, and C with some values (example: 1.0 + 0.0i, etc.)
    for (size_t i = 0; i < total_elements_A; ++i) h_A[i] = make_complex(1.0, 0.0);
    for (size_t i = 0; i < total_elements_B; ++i) h_B[i] = make_complex(0.5, 0.0);
    // Initialize C with ones so we can test the beta scaling
    for (size_t i = 0; i < total_elements_C; ++i) h_C[i] = make_complex(1.0, 0.0); 

    // Initialize the CPU verification array with the same start values as h_C
    h_C_cpu_verify = h_C;

    // Scaling factors: C = 1.0 * A * B + 1.0 * C 
    hipDoubleComplex alpha = make_complex(1.0, 0.0);
    hipDoubleComplex beta  = make_complex(1.0, 0.0);

    // --- 2. Device memory allocation and data transfer ---
    hipDoubleComplex *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, total_elements_A * sizeof(hipDoubleComplex)));
    CHECK_HIP(hipMalloc(&d_B, total_elements_B * sizeof(hipDoubleComplex)));
    CHECK_HIP(hipMalloc(&d_C, total_elements_C * sizeof(hipDoubleComplex)));

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), total_elements_A * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), total_elements_B * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C, h_C.data(), total_elements_C * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));

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
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<hipDoubleComplex> h_C_gpu_result(total_elements_C);
    CHECK_HIP(hipMemcpy(h_C_gpu_result.data(), d_C, total_elements_C * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));

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
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
 }
