/**********************************
 **  This is a code by Gemini and I 
 ** 
 ***/


#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>


/// Miscellaneous errors  

#define CHECK_HIP_ERROR(call) do {		\
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





/******************
 *  STEP 1: Projection
 *  P = V^t * H * V
 ***/ 


static inline
void projection(rocblas_handle handle,
		Matrix   &H, Matrix   &V, Matrix   &T, Matrix &P,
		double *dH, double *dV, double *dT, double *dP, 
		) {
  
  // T = H * V
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_none, rocblas_operation_none, 
		      H.m, V.n, H.n,  // this is the problem size                                                 
		      &alpha, 
		      dH, H.m, 
		      dV, V.m, 
		      &beta, 
		      dT, T.m)); 

  // P = V^t T 
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_transpose, // transpose 
		      rocblas_operation_none, 
		      P.m, T.n, T.m, // this is the problem size
		      &alpha, 
		      dV, V.m, 
		      dT, T.m, 
		      &beta, 
		      dP, P.m)); 
  

}

/******************
 *  STEP 2: Eigensolver 
 *  A -> A,W
 *       A = Matrix of Eigenvectors 
 *       W = Eigenvalues
 *   
 ***/ 

static inline
int eigenSolver(rocblas_handle handle,
		 Matrix &A,
		 double *dA, double *dW, double *dWW,  int *info_device) {

  int info_host = 0;

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

  check_hip_error(hipMemcpy(&info_host, info_device, sizeof(int), hipMemcpyDeviceToHost));
  
  // 6. Check for convergence issues and print results
  if (info_host == 0) {
    return 1;
  } else {
    return 0;
    std::cerr << "The algorithm failed to converge. info = " << info_host << std::endl;
  }
}

/***************
 * STEP 3: sort the eigenvectors by eigenvalues
 * A : Eigenvector Matrix
 * d_sort_keys : eigenvalues 
 * B sorted eigenvectors 
 * d_perm_indices : temporary space for the permutation 
 */

extern void sortarg(int N, // number of rows 
		    int M, // number of keys  or columns 
		    double *d_A, // source matrix  N rows and M columns we swap the columns
		    double *d_B, // permuted destination matrix 
		    double *d_sort_keys, // sorting keys
		    int* d_perm_indices // permutation 
		    );

/******************
 * STEP 4 : V = V* B above Ritz vectors
 */


/******************
 * STEP 5 : Computation of the residual to see if we can stop the
 * iterations
 */

extern void residuals(rocblas_handle handle,
		      int M, int N_EIG,
		      double *d_H, 
		      double *d_X, // Eigenvectors 
		      double *d_eig_vals, // eigenvalues 
		      double *d_HX_inter, // intermediate result
		      double *d_R  // residuals
		      );


/****************
 * STEP 6: norm of the residuals 
 * < tolerance we are done 
 */

std::vector<double>
calculate_norm(
	       int N, 
	       int M,
	       double* d_A,
	       double* d_norms,
	       rocblas_handle handle) {
  
  std::vector<double> h_norms(N);
  
  // Set pointer mode to device, as results are stored on the device
  ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
  
  // --- rocBLAS Call (Strided Batched NRM2) ---
  std::cout << "Calling rocblas_dnrm2_strided_batched..." << std::endl;
  ROCBLAS_CHECK(
	 rocblas_dnrm2_strided_batched(
	         handle,
		 M,               // n: length of each vector
		 d_A,             // x: device pointer to the matrix/vectors
		 1,               // incx: stride within each vector (1 for column-major matrix)
		 M,               // stride_x: stride between consecutive vectors in memory
		 N,               // batch_count: number of vectors (columns)
		 d_norms          // result: device pointer to store norms
				       ));

    // --- Copy results back to host ---
    CHECK_HIP_ERROR(hipMemcpy(h_norms.data(), d_norms, N * sizeof(double), hipMemcpyDeviceToHost));
    return h_norms
}



/****************
 * STEP 7: Not converged we compute the corrections 
 */

extern corrections(int M, int N_EIG,
		   double *d_R,      // residuals 
		   double *d_diag_H, // H diagonal
		   double *d_eig_vals, // eigenvalues 
		   double *d_T,         // result
		   const double epsilon); 

/***************
 * STEP 8:  V  = V+  Corrections
 */



/***************
 * STEP 9:  V,R = QR(V)
 */

static inline
void QR(rocblas_handle handle,
	Matrix &A,    // the original matrix in the CPU
	double *dA,  // this will be a pointer in the GPU of the Q of QR 
	double *dTau // we need the pointer 
	) {
  
  CHECK_ROCSOLVER_STATUS(rocsolver_dgeqrf(handle, A.m, A.n, dA, A.m /*LDA column major*/, dTau));
  CHECK_ROCSOLVER_STATUS(rocsolver_dorgqr(handle, A.m, A.n, std::min( A.m, A.n), dA, A.m, dTau));
  
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







#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>

// ... (HIP_CHECK and ROCBLAS_CHECK macros remain the same) ...




		      

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
`
  argsortCpp(diag);
  
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

    // We copy the new subspace V into GPU  
    CHECK_HIP_ERROR(hipMemcpy(dV,  V.matrix.data(), V.size() * sizeof(double), hipMemcpyHostToDevice));


    // P <- V^t * H * V projection into a smaller space -> dP is in
    // Python: T = V.T @ A @ V 
    // the GPU
    projection(handle, H,V,T,P, dH,dV,dT,dP); 

    
    // dW has the sorted eigenvalue and dP has the sorted eigenvector 
    // Python: eig_vals_T, eig_vecs_T = np.linalg.eigh(T) # rocsolver_dsyev_ 
    eigenSolver(handle,P, dP, dW, dWW,info_device); 

    /* Python: 
       idx = np.argsort(eig_vals_T)
       eig_vals_T = eig_vals_T[idx]
       eig_vecs_T = eig_vecs_T[:, idx]
    */
    sortarg(P.m,  num_eigs /*P.n*/, dP, dSP, dW, d_perm_indices); // dPP is the sorted eigen vectors 

    
    // V * EigenVectors Called Ritz vectors
    // Python: current_eig_vecs = V @ eig_vecs_T[:, :num_eigs]
    CHECK_ROCBLAS_STATUS(
	   rocblas_dgemm(
	     handle, 
	     rocblas_operation_none, 
	     rocblas_operation_none, 
	     V.m, V.n, num_eigs, // notice the last dimension
	     &alpha, 
	     dV, V.m, // lead V
	     dSP,P.m, // lead EigenVectors 
	     &beta,   
	     dP, P.m  // lead P we reuse the projection
	   )
     );   

    residuals() ;
    
      

    
    std::vector<double> norms = calculate_norm(num_eigs,H.m,
					       dX,
					       d_norms,
					       handle);
    int converged_count = 0
    for ( int i=0; i< num_eigs; i++) 
      if (norms[i] < tolerance):
	converged_count += 1;


    if (converged_count == num_eigs) {
      printf("Davidson converged after %d iteration", iteration);
      CHECK_HIP_ERROR(hipMemcpy(eigenvale.data(), dW, P.n * sizeof(double), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(eigenvector.data(), dP, P.size() * sizeof(double), hipMemcpyDeviceToHost));
      break;
    }
    
    corrections();
    qr()
  }
}
    
  

int main(int argc, char* argv[]) {
  
  // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = std::atoi(argv[1]);
  std::cout << "Integer from atoi: " << mydevice << std::endl;
  int result = set_device(mydevice);
  int M = std::atoi(argv[2]);;
  int n_eng =std::atoi(argv[4]);

  std::vector<double> hH(M * M);    
  Matrix H = {hH, M,M};
  
  davidson_rocm(H,
		n_eigs,
		100,
		1e-8);

  
}


