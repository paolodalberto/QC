#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK_HIP_ERROR(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " : " \
                  << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_ROCBLAS_STATUS(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
      std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__  << " : " \
		<< rocblas_status_to_string(status) << " : "<< std::endl;	\
        exit(EXIT_FAILURE); \
    } \
} while (0)




struct matrix {
  std::vector<double> &matrix; 
  int m;  // rows
  int n;  // cols
  // Host-side matrix multiplication for verification (optional)
  void (*gemm)(struct matrix &C, double beta, struct matrix &A, struct matrix &B, double alpha);
  void init() {
    for (int i = 0; i < m * n; ++i) matrix[i] = static_cast<double>(i % 10);
  };
  void zero() {
    for (int i = 0; i < m * n; ++i) matrix[i] = static_cast<double>(0);
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
    printf("Column Major M,N = %d,%d \n", m,n);
    if (t)
	for (int i = 0; i < MM; ++i) {
	  for (int j = 0; j < NN; ++j) {
	    printf("%0.1f ", matrix[ind(i,j)]);
	}
	printf("\n");
      }
  };
};
typedef struct matrix Matrix; 

void cpu_dgemm(Matrix &C, double beta, Matrix &A, Matrix &B, double alpha) {
  int M = A.m;
  int N = B.n;
  int K = A.n;
  
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int l = 0; l < K; ++l) {
	sum += A.matrix[A.ind(i,l)] * B.matrix[B.ind(l,j)];
      }
      C.matrix[C.ind(i,j)] = alpha * sum + beta * C.matrix[C.ind(i,j)];
    }
  }
}

void cpu_dgemm_At(Matrix &C, double beta, Matrix &A, Matrix &B, double alpha) {
  int M = C.m;
  int N = B.n;
  int K = B.m;
  
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int l = 0; l < K; ++l) {
	sum += A.matrix[A.ind(l,i)] * B.matrix[B.ind(l,j)];
      }
      C.matrix[C.ind(i,j)] = alpha * sum + beta * C.matrix[C.ind(i,j)];
    }
  }
}


int set_device(int id) {
  int deviceCount;
  CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return -1;
  }

  // Enumerate devices and their properties (optional, but good practice)
  for (int i = 0; i < deviceCount; ++i) {
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, i));
    std::cout << "Device " << i << ": " << props.name << std::endl;
  }

  // Set the desired device (e.g., device 0)
  int desiredDevice = id; 
  if (desiredDevice < deviceCount) {
    CHECK_HIP_ERROR(hipSetDevice(desiredDevice));
    std::cout << "Successfully set device to " << desiredDevice << std::endl;
  } else {
    std::cerr << "Invalid device index: " << desiredDevice << std::endl;
    return -1;
  }
  return id;
}

  

int main(int argc, char* argv[]) {
  
  // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = std::atoi(argv[1]);
  std::cout << "Integer from atoi: " << mydevice << std::endl;
  int result = set_device(mydevice);
  int M = std::atoi(argv[2]);;
  int N = std::atoi(argv[3]);
  int Tt =std::atoi(argv[4]);
  
  
  
    double alpha = 1.0;
    double beta = 0.0;

    // we are going to compute the projection V^t * H * V
    std::vector<double> hH(M * N);    
    std::vector<double> hV(M * Tt);
    std::vector<double> hP(Tt * Tt);
    std::vector<double> hT(M * Tt);

    
    Matrix H = {hH, M,N,cpu_dgemm};
    Matrix V = {hV, M,Tt,cpu_dgemm};
    Matrix P = {hP, Tt,Tt,cpu_dgemm};
    Matrix T = {hT, M,Tt,cpu_dgemm};
        

    // Allocate host memory (using double)

    // Initialize host matrices
    H.init();
    V.init();
    T.zero();
    P.zero();

    

    // Allocate device memory (using double)
    double *dP, *dH, *dV, *dT;
    CHECK_HIP_ERROR(hipMalloc(&dP, P.size()* sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dT, T.size()* sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dH, H.size()* sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dV, V.size()* sizeof(double)));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(dH, H.matrix.data(), H.size() * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dV, V.matrix.data(), V.size() * sizeof(double), hipMemcpyHostToDevice));

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    printf(" H \n");
    H.print(false);
    printf(" V \n");
    V.print(false);
    cpu_dgemm(T,beta, H,V,alpha);
    printf(" CPU H*V \n");
    T.print(true);

    
    CHECK_ROCBLAS_STATUS(
	  rocblas_dgemm(handle, 
			rocblas_operation_none, rocblas_operation_none, 
			H.m, V.n, H.n,                                                  
			&alpha, 
			dH, H.m, 
			dV, V.m, 
			&beta, 
			dT, T.m)); 

    T.zero();
    CHECK_HIP_ERROR(hipMemcpy(T.matrix.data(), dT, T.size() * sizeof(double), hipMemcpyDeviceToHost));
    printf(" GPU T \n");
    T.print(true);

    printf(" V^T * T \n ");
    V.print(false);
    T.print(false);
    // P = V^t T 
    CHECK_ROCBLAS_STATUS(
	  rocblas_dgemm(handle, 
			rocblas_operation_transpose, rocblas_operation_none, 
			P.m, T.n, T.m,                                                  
			&alpha, 
			dV, V.m, 
			dT, T.m, 
			&beta, 
			dP, P.m)); 

    // Copy result back to host
    CHECK_HIP_ERROR(hipMemcpy(P.matrix.data(), dP, P.size() * sizeof(double), hipMemcpyDeviceToHost));
    printf("GPU \n");
    P.print(true);
    P.zero();
    cpu_dgemm_At(P,beta, V,T,alpha);
    printf("CPU \n");
    P.print(true);
      
    // Destroy rocBLAS handle and free device memory
    CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));
    CHECK_HIP_ERROR(hipFree(dH));
    CHECK_HIP_ERROR(hipFree(dV));
    CHECK_HIP_ERROR(hipFree(dP));
    CHECK_HIP_ERROR(hipFree(dT));





    
    std::cout << "DGEMM operation complete using rocBLAS." << std::endl;
    return 0;
}
