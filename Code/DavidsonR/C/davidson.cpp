


#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>


extern void sortarg(int N, // number of rows 
		    int M, // number of keys  or columns 
		    double *d_A, // source matrix  N rows and M columns we swap the columns
		    double *d_B, // permuted destination matrix 
		    double *d_sort_keys, // sorting keys
		    int* d_perm_indices // permutation 
		    );



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
	Matrix &A,    // the original matrix in the CPU
	double *dA,  // this will be a pointer in the GPU of the Q of QR 
	double *dTau // we need the pointer 
	) {
  
  CHECK_ROCSOLVER_STATUS(rocsolver_dgeqrf(handle, A.m, A.n, dA, A.m /*LDA column major*/, dTau));
  CHECK_ROCSOLVER_STATUS(rocsolver_dorgqr(handle, A.m, A.n, std::min( A.m, A.n), dA, A.m, dTau));
  
}

static inline
void projection(rocblas_handle handle,
		Matrix   &H, Matrix   &V, Matrix   &T, Matrix &P,
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
void eigenSolver(rocblas_handle handle,
		 Matrix &A,
		 double *dA, double *dW, double *dWW,  int *info_device) {


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





/**
 * Kernel for element-wise multiplication with broadcasting.
 * Assumes column-major storage for the matrix.
 * Vector 'd_a' is broadcast across the columns of matrix 'd_b'.
 */
__global__ void elementwise_mult_broadcast(int N, int M, 
                                           const double* d_a, // Vector [M elements]
                                           const double* d_b, // Matrix [N*M elements]
                                           double* d_c)       // Output Matrix [N*M elements]
{
    // Calculate global thread ID in X and Y dimensions (row and column)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        // Index calculation for column-major storage: index = row + col * N
        
        double vector_val = d_a[col];      // The value to broadcast for this column
        double matrix_val = d_b[row + col * N]; // The matrix element
        
        d_c[row + col * N] = matrix_val * vector_val;
    }
}

void vector_by_natrix(int N, int M, 
		      const double* d_a, // Vector [M elements]
		      const double* d_b, // Matrix [N*M elements]
		      double* d_c) {     // destination

  // 4. Launch the kernel
  // We launch a 2D grid of threads (N rows * M columns is a safe size)
  dim3 blocks( (N + 15) / 16, (M + 15) / 16 ); // Example block size calculation
  dim3 threadsPerBlock(16, 16); 
  
  elementwise_mult_broadcast<<<blocks, threadsPerBlock>>>(
							  N, M, d_a, d_b, d_c
							  );



}
  



		      

struct Value_Index {
  double value;
  int    index;
};

// Custom comparator functor for descending order
struct CompareFunction {
  __device__ bool operator()(const Value_Index& a, const Value_Index& b) const {
    return a.value < b.value; // Use 'a > b' for descending order
  }
};



static inline 
void argsortCpp( std::vector<Value_Index> &A ) {

  std::sort(A.begin(), A.end(), CompareFunction());


}






// Matrices are in column majors 

void davidson_rocm( Matrix  H,
		    int num_eigs,
		    int max_iterations=100,
		    double tolerance=1e-8) {


  // Create rocBLAS handle
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

  int M = std::min(num_eigs*max_iterations, H.n) // H is square? 

  std::vector<double> vdata[H.m * M];
  std::vector<double> pdata[M * M];
  std::vector<double> edata[M];
  
  
  Matrix V = {vdata,H.m, num_eigs};
  Matrix P = {pdata,V.n,V.n};
  Matrix E = {edata, num_eigs,1};
  Matrix T = {{}, H.m,V.n};


  V.zero();

  // This is what should be 
  
  //  # Initial guess vectors (e.g., identity matrix columns corresponding to lowest diagonal elements)
  //  # Sort diagonal indices to pick the ones corresponding to the lowest expected eigenvalues
  //  diag_indices = np.argsort(np.diag(A))
  //  V = np.zeros((n, num_eigs))
  //  for i in range(num_eigs):
  //      V[diag_indices[i], i] = 1.0


  std::vector<Struct Value_Index> diag[H.n];
  Matrix ES = {diag, num_eigs,1};

  for (int i=0;i<H.m, i++) {
    diag[i].value = H.matrix[H.ind(i,i)];
    diag[i].index = i;
  }

  sargsortCpp(diag);
  
  for (int i=0;i<num_eigs, i++) {
    V.matrix[V.ind(diag[i].index,i)] = 1;
  }

  // No need to QR a unitary set of vectors ...
  
  // Copy data from host to device once because this will not change.
  double *dP, *dSP, *dH, *dV, *dT;
  double *dW, *dWW;
  int *info_device;
  double alpha = 1.0;
  double beta = 0.0;
  
  
  CHECK_HIP_ERROR(hipMalloc(&dH, H.size()* sizeof(double)));
  CHECK_HIP_ERROR(hipMemcpy(dH, H.matrix.data(), H.size() * sizeof(double), hipMemcpyHostToDevice));

  // We allocate the largest V we can afford once
  CHECK_HIP_ERROR(hipMalloc(&dV, H.m*M* sizeof(double)));
  // We allocate the largest P = V^t * H * V we can afford once
  CHECK_HIP_ERROR(hipMalloc(&dP, M*M* sizeof(double)));         // Projection -> eigenvectors  
  CHECK_HIP_ERROR(hipMalloc(&dSP, M*num_eigs* sizeof(double))); // sorted eigenvectors
  // We allocate the largest T = H*V we can afford once
  CHECK_HIP_ERROR(hipMalloc(&dT, H.m*M* sizeof(double)));

  // 2. Allocate device memory
  CHECK_HIP_ERROR(hipMalloc(&W_device, sizeof(double) * M));    // Eigen value
  CHECK_HIP_ERROR(hipMalloc(&work_device, sizeof(double) * M)); // work space 
  CHECK_HIP_ERROR(hipMalloc(&info_device, sizeof(int)));        // solver burfing 

   
  for (int iteration=0; iteration<max_iterations; iterations++) {

    // these are the temporary matrices for the projections that are
    // allive only in the GPU 
    P.m= P.n =  V.n;
    T.m = V.n;

    // we copy the new subspace 
    CHECK_HIP_ERROR(hipMemcpy(dV,  V.matrix.data(), V.size() * sizeof(double), hipMemcpyHostToDevice));

    projection(handle, H,V,T,P, dH,dV,dT,dP); // P <- V^t * H * V 

    eigenSolver(handle,P, dP, dW, dWW,info_device); // dP has now the eigenvectors and dW the eigenvalue

    sortarg(P.m,  num_eigs /*P.n*/, dP, dPP, dW, d_perm_indices); // dP is the sorted eigen vectors 
    
    // V * EigenVectors Called Ritz vectors 
    CHECK_ROCBLAS_STATUS(
	   rocblas_dgemm(handle, 
			 rocblas_operation_none, rocblas_operation_none, 
			 V.m, V.n, num_eigs,                                                  
			 &alpha, 
			 dV, V.m, 
			 dSP,P.m, 
			 &beta, 
			 dP, P.m));   

    
    //  ZGEMM  and ZGEMA 
    //  residuals = A @ current_eig_vecs -   current_eig_vals * current_eig_vecs

    
    CHECK_ROCBLAS_STATUS(
	   rocblas_dgemm(handle, 
			 rocblas_operation_none, rocblas_operation_none, 
			 H.m, P.n,num_eigs,                                                  
			 &alpha, 
			 dH, H.m, 
			 dP,P.m, 
			 &beta, 
			 dX, X.m)); 


    
    
    
    
    
    
    

    
      






    
  }
    
  





































}


