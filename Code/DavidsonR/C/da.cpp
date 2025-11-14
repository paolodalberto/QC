


#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#define CHECK_HIP_ERROR(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) {						\
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " : " \
		<< hipGetErrorString(err) << std::endl;			\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

#define CHECK_ROCSOLVER_STATUS(call)  {	    \
    rocblas_status status = call;	    \
    if (status != rocblas_status_success) {				\
      std::cerr << "rocSOLVER error at " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(EXIT_FAILURE);						\
    }									\
  }



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


static inline
void QR(rocblas_handle handle,
	Matrix A,    // the original matrix in the CPU
	double *dA,  // this will be a pointer in the GPU of the Q of QR 
	double *dTau // we need the pointer 
	) {
  
  CHECK_ROCSOLVER_STATUS(rocsolver_dgeqrf(handle, A.m, A.n, dA, A.m /*LDA column major*/, dTau));
  CHECK_ROCSOLVER_STATUS(rocsolver_dorgqr(handle, A.m, A.n, std::min( A.m, A.n), dA, A.m, dTau));
  
}

static inline
void Projection(rocblas_handle handle,
		Matrix   H, Matrix   V, Matrix   T, Matrix P,
		double *dH, double *dV, double *dT, double *dP, 
		) {
  
  // T = H * V
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_none, rocblas_operation_none, 
		      H.m, V.n, H.n,                                                  
		      &alpha, 
		      dH, H.m, 
		      dV, V.m, 
		      &beta, 
		      dT, T.m)); 

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
  

}

static inline
void EigenSolver(rocblas_handle handle,
		 Matrix A,
		 double *dA, double *dW, double *dWW,  int *info_device = 0) {


  CHECK_ROCSOLVER_STATUS(
        rocsolver_dsyev(
			handle,
			rocblas_evect_original, //rocblas_evect_full, // Jobz: use the enum value
			rocblas_fill_upper, // Uplo: use the enum value
			A.n,                  
			dA,           
			n,                  
			dW,
			dWW,
			info_device         
			));


}
			       



void davidson_rocm( Matrix  H,
		    int num_eigs,
		    int max_iterations=100,
		    double tolerance=1e-8) {


  // Create rocBLAS handle
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));




  std::vector<double> vdata[num_eigs*num_eigs];
  Matrix V = {num_eigs, num_eigs,vdata};
  V.zero();
  
  std::vector<double> diag[H.n];
  for (int i=0;i<H.m, i++) {
    diag[i] = H.matrix[i*n+i];
  }

  std::sort(diag, diag + H.n);
  
  for (int i=0;i<num_eigs, i++) {
    V.matrix[i*n+i] = diag[i];
  }


  
  // Copy data from host to device

  CHECK_HIP_ERROR(hipMalloc(&dH, H.size()* sizeof(double)));
  CHECK_HIP_ERROR(hipMemcpy(dH, H.matrix.data(), H.size() * sizeof(double), hipMemcpyHostToDevice));
  

  
  
  for (int iteration=0; iteration<max_iterations; iterations++) {
    CHECK_HIP_ERROR(hipMemcpy(dV, V.matrix.data(), V.size() * sizeof(double), hipMemcpyHostToDevice));

    Matrix P = {{}, V.n,V.n,cpu_dgemm};
    Matrix T = {{}, M,Tt,cpu_dgemm};
  
  
    double *dP, *dH, *dV, *dT;
    CHECK_HIP_ERROR(hipMalloc(&dP, P.size()* sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dT, T.size()* sizeof(double)));
    
    CHECK_HIP_ERROR(hipMalloc(&dV, V.size()* sizeof(double)));
  



    
    Projection();
    EigenSolver();
    
    argsort();

    
      






    
  }
    
  





































}


