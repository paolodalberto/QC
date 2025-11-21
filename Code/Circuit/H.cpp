typedef rocblas_double_complex ZC;

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
      if (host)   matrix = (double*) std::calloc(M*N,sizeof(ZC));
      if (device) CHECK_HIP_ERROR(hipMalloc(&d_matrix, M*N* sizeof(ZC)));		      
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
  Matrix I; // Input state : 2^n x1 
  Matrix U; // Gate matrix gxg 
  int    I_l; // batch
  int    I_r; // outer dimension of I_l(x)G(x)I_r
  Matrix O; // output state 

};





#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas.h>

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

int main() {
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // Matrix dimensions (all matrices in the batch must have the same dimensions)
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int k = 2;
    rocblas_int lda = m;
    rocblas_int ldb = k;
    rocblas_int ldc = m;
    rocblas_int batch_count = 3;

    // Scalars
    ZC alpha = {1.0, 0.0}; // alpha = 1.0 + 0.0i
    ZC beta = {0.0, 0.0};  // beta = 0.0 + 0.0i

    // Host data (example values)
    std::vector<ZC> hA_data = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, // Matrix A1 2x2
        {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}, // Matrix A2 2x2
        {9.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}  // Matrix A3 2x2
    };
    std::vector<ZC> hB_data = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, // Matrix B1 2x2
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, // Matrix B2 2x2
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}  // Matrix B3 2x2
    };
    std::vector<ZC> hC_data(m * n * batch_count, {0.0, 0.0});

    // Allocate device memory for data
    ZC *dA_data, *dB_data, *dC_data;
    CHECK_HIP_ERROR(hipMalloc(&dA_data, hA_data.size() * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&dB_data, hB_data.size() * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&dC_data, hC_data.size() * sizeof(ZC)));

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(dA_data, hA_data.data(), hA_data.size() * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB_data, hB_data.data(), hB_data.size() * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_data, hC_data.data(), hC_data.size() * sizeof(ZC), hipMemcpyHostToDevice));

    // Create device pointers for the array of matrix pointers
    ZC **d_A_array, **d_B_array, **d_C_array;
    CHECK_HIP_ERROR(hipMalloc(&d_A_array, batch_count * sizeof(ZC*)));
    CHECK_HIP_ERROR(hipMalloc(&d_B_array, batch_count * sizeof(ZC*)));
    CHECK_HIP_ERROR(hipMalloc(&d_C_array, batch_count * sizeof(ZC*)));

    // Prepare host-side array of pointers
    std::vector<ZC*> h_A_array(batch_count);
    std::vector<ZC*> h_B_array(batch_count);
    std::vector<ZC*> h_C_array(batch_count);
    size_t matrix_size = m * n * sizeof(ZC); // For A and C
    size_t b_matrix_size = k * n * sizeof(ZC); // For B

    for (int i = 0; i < batch_count; ++i) {
        h_A_array[i] = dA_data + i * (m * k);
        h_B_array[i] = dB_data + i * (k * n);
        h_C_array[i] = dC_data + i * (m * n);
    }

    // Copy array of pointers to device memory
    CHECK_HIP_ERROR(hipMemcpy(d_A_array, h_A_array.data(), batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B_array, h_B_array.data(), batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C_array, h_C_array.data(), batch_count * sizeof(ZC*), hipMemcpyHostToDevice));

    // Call rocblas_zgemm_batched
    CHECK_ROCBLAS_ERROR(rocblas_zgemm_batched(
        handle,
        rocblas_operation_none, rocblas_operation_none, // No transpose for A and B
        m, n, k,
        &alpha,
        (const ZC* const*)d_A_array, lda,
        (const ZC* const*)d_B_array, ldb,
        &beta,
        d_C_array, ldc,
        batch_count
    ));

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(hC_data.data(), dC_data, hC_data.size() * sizeof(ZC), hipMemcpyDeviceToHost));

    // Print results (optional)
    std::cout << "Batched ZGEMM results:" << std::endl;
    for (int i = 0; i < batch_count; ++i) {
        std::cout << "Matrix C" << i + 1 << ":" << std::endl;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                int idx = i * (m * n) + row * n + col; // Simple row-major index for printing
                std::cout << "(" << hC_data[idx].real() << ", " << hC_data[idx].imag() << ") ";
            }
            std::cout << std::endl;
        }
    }

    // Cleanup
    CHECK_HIP_ERROR(hipFree(dA_data));
    CHECK_HIP_ERROR(hipFree(dB_data));
    CHECK_HIP_ERROR(hipFree(dC_data));
    CHECK_HIP_ERROR(hipFree(d_A_array));
    CHECK_HIP_ERROR(hipFree(d_B_array));
    CHECK_HIP_ERROR(hipFree(d_C_array));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
