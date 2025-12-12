


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
#include "davidson.h"




/**********
 * Reference computation in basic c++
 * using regular pointers
 *
 */ 

extern
void cpu_zgemm_batched_b(int M, int N, int K, ZC alpha, 
		       ZC* A, rocblas_stride ldA,
		       ZC* B, rocblas_stride ldB,
		       ZC beta,
		       ZC* C, rocblas_stride ldC,
		       int batchCount
		       );
/***
 * For batched computation we need to prepare the pointers
 */

extern 
void pre_gpu_gemm(
		  int M, int N, int K, 
		  ZC** A, rocblas_stride ldA, ZC *d_A,
		  ZC** B, rocblas_stride ldB, ZC *d_B,
		  ZC** C, rocblas_stride ldC, ZC *d_C,
		  int batchCount
		  );
/***
 * After the pre you can run the execute 
 */
extern 
void gpu_zgemm_batched(
		       rocblas_handle handle,
		       int M, int N, int K, ZC alpha, 
		       ZC** A, rocblas_stride ldA,
		       ZC** B, rocblas_stride ldB,
		       ZC beta,
		       ZC** C, rocblas_stride ldC,
		       int batchCount
		       );




/*****
 * this is column major and thus we will need to compute O = I * G^t
 * but I will transpose directly G  ...
 ***/

void cpu_zgemm_batched_M(
     int M, int N, int K, ZC alpha, 
     Matrix &A,
     Matrix &B,  // B is the small one the gate one 
     ZC beta,
     Matrix &C,
     int batchCount) {
  
  cpu_zgemm_batched_b(
	   A.m, B.m, A.n, alpha, 
	   A.matrix, A.m, // I 
	   B.matrix, B.m, // G^t
	   ZC beta,
	   C.matrix, C.m, // O
	   int batchCount
		      );
  
}





/***************
 **  A circuit is a sequence of layers. This define the depth of the
 **  circuit. The layers is a sequence gates.  

 **  Each gate has the property that the order does not matter but
 **  they are NOT parallel.  Again take a look at the notes and the
 **  python implementation. There is an opportunity to fuse Gates to
 **  reduce the number of passes throught the state but this will
 **  affect the "rows" of the computation and thus the interference in
 **  caches.
 **
 **
 ********/



struct gate {

  /* Every one deserves a unique name so if you are
   using the same gate identified by a name we
   can do this quite easily and we do not need to
   have multiple copies .. there will be only one
   gate but it will be applied to different bits
   thus the computation will be different
  */
  std::string name;  


  int    bit_number; // the first bit where we apply the gate k
                   

  Matrix &I; // Input state : 2^n x 1: shared 
  Matrix &U; // Gate matrix gxg      : shared and already transposed   
  Matrix &O; // output state         : shared and this can be the input state  

  int m=0; // index of there we apply the gate
  int n=0; // kernel size and this is G kernel (mxk)
  int k=0;

  int batch_count =1;
  ZC alpha = ALPHA;
  ZC beta  = BETA; 

  // host pointers the strided pointers for the computation in the host 
  ZC **h_A_ptrs =0 ;
  ZC **h_B_ptrs =0 ;
  ZC **h_C_ptrs =0 ;

  // device pointers  as above 
  ZC **d_A_ptrs = 0 ;
  ZC **d_B_ptrs =0 ;
  ZC **d_C_ptrs =0;


  
  // We allocate the pointers only  
  void alloc(bool host, bool device) {
    if (host) { 
      h_A_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_A_ptrs did not make it");
      h_B_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_B_ptrs did not make it");
      h_C_ptrs = (ZC**)malloc(batchCount * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_C_ptrs did not make it");
    }
    if (device ) {
      CHECK_HIP(hipMalloc((void**)&d_A_ptrs, batchCount * sizeof(ZC*)));
      CHECK_HIP(hipMalloc((void**)&d_B_ptrs, batchCount * sizeof(ZC*)));
      CHECK_HIP(hipMalloc((void**)&d_C_ptrs, batchCount * sizeof(ZC*)));
    }
  }

  // we free the pointers and the gate
  void free() {
    
    if (h_A_ptrs) { free(h_A_ptrs); h_A_ptrs=0;} 
    if (h_B_ptrs) { free(h_B_ptrs); h_B_ptrs=0;} 
    if (h_C_ptrs) { free(h_C_ptrs); h_C_ptrs=0;} 

    CHECK_HIP_ERROR(hipFree(d_A_ptrs));
    CHECK_HIP_ERROR(hipFree(d_B_ptrs));
    CHECK_HIP_ERROR(hipFree(d_C_ptrs));

    // gate
    U.free();
  }

  
  void init() {
    // U is square and we allocate in the host and the device
    U.alloc(true, true);

    // remember we are doing O = I* U (where U is U^t)
    
    int B = I.m; // this is the state 2^n 
    batch_count = B - ((1<<bit_number)+U.m);

    m = 1<<bit_number;
    n = U.n; 
    k = U.m;

    alloc(true,true)
    pre_gpu_gemm(m,n,k,
		 h_A_ptrs,m,U.matrix,
		 h_B_ptrs,ldB,I.matrix,
		 h_C_ptrs,ldC,O.matrix,
		 batchCount);

  }
  
  
  
  void step(rocblas_handle handle) {

    const rocblas_stride strideA = m*n;
    const rocblas_stride strideB = k*n;
    const rocblas_stride strideC = n * n;
    const size_t total_elements_A = strideA * batch_count;
    const size_t total_elements_B = strideB * batch_count;
    const size_t total_elements_C = strideC * batch_count;


    CHECK_ROCBLAS_ERROR(
	gpu_zgemm_batched(
	    handle,
	    m, n, k, alpha, 
	    d_A_ptrs, strideA,
	    d_B_ptrs, strideB,
	    beta,
	    d_C_ptrs, strideC,
	    batchCount
			  );
  
};

typedef struct gate Gate;

struct schedule {
  
  Matrix &I;  // Input state 
  Matrix &O;  // Output state
  std::vector<std::vector<Gate>> schedule; 

  // we move all the matrices into the 
  void init(){
    for (std::vector<Gate> &level  : schedule)
      for (Gate h : level )
	h.init();
    
  }
  
  
  void forward(rocblas_handle handle) {
    for (std::vector<Circuit> &level  : schedule)
      for (Circuit h : level )
	h.step(handle);
  }
};

typedef struct schedule Circuit;





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
