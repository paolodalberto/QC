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

#define CHECK_ROCBLAS_STATUS(call)  {					\
    rocblas_status status = call;					\
    if (status != rocblas_status_success) {				\
      std::cerr << "rocm error at " << __FILE__ << ":" << __LINE__ << ":" \
		<< rocblas_status_to_string(status) << std::endl;		\
      exit(EXIT_FAILURE);						\
    }									\
  }

// I like to wrap the matrices into a container

struct matrix {

  // struct as a class members are all public
  double *matrix; // host  
  int m;                       // rows
  int n;                       // cols
  double *d_matrix;            // device


  // functions 
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



void GEMMP(struct matrix &C, double beta, struct matrix &A, struct matrix &B, double alpha, rocblas_handle handle=0) {

  // T = H * V
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_none, rocblas_operation_none, 
		      A.m, B.n, B.n,  // this is the problem size                                                 
		      &alpha, 
		      A.d_matrix, A.m, 
		      B.d_matrix, B.m, 
		      &beta, 
		      C.d_matrix, C.m)); 

}


/******************
 *  STEP 1: Projection
 *  P = V^t * H * V
 ***/ 



static inline
void projection(rocblas_handle handle,
		Matrix   &H, Matrix   &V, Matrix   &T, Matrix &P,
		double alpha = 1.0,  double beta = 0.0
) {


  printf("H \n"); H.print();
  printf("V \n"); V.print();
  printf("T \n"); T.print();
  // T = H * V
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_none, rocblas_operation_none, 
		      H.m, V.n, H.n,  // this is the problem size                                                 
		      &alpha, 
		      H.d_matrix, H.m, 
		      V.d_matrix, V.m, 
		      &beta, 
		      T.d_matrix, T.m)); 

  printf("P \n"); P.print();
  // P = V^t T 
  CHECK_ROCBLAS_STATUS(
	rocblas_dgemm(handle, 
		      rocblas_operation_transpose, // transpose 
		      rocblas_operation_none, 
		      P.m, T.n, T.m, // this is the problem size
		      &alpha, 
		      V.d_matrix, V.m, 
		      T.d_matrix, T.m, 
		      &beta, 
		      P.d_matrix, P.m)); 
  

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
		Matrix &EVa,
		Matrix &Work,
		int *info_device) {

  int info_host = 0;

  CHECK_ROCBLAS_STATUS(
        rocsolver_dsyev(
			handle,
			rocblas_evect_original, //rocblas_evect_full, // Jobz: use the enum value
			rocblas_fill_upper, // Uplo: use the enum value
			A.n,                  
			A.d_matrix,           
			A.n,                  
			EVa.d_matrix,
			Work.d_matrix,
			info_device         
			));

  CHECK_HIP_ERROR(hipMemcpy(&info_host, info_device, sizeof(int), hipMemcpyDeviceToHost));
  
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

void sortarg(Matrix &EVe, Matrix &EVe_sorted, Matrix &EVa,  int* d_perm_indices)  {
  return  sortarg(EVe.m, EVe.n, EVe.d_matrix, EVe_sorted.d_matrix, EVa.d_matrix,d_perm_indices);
}

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

void residuals(rocblas_handle handle,
	       Matrix &H, Matrix &X, Matrix &EVa, Matrix &HX_inter, Matrix &R, 
	       int N_EIG) {
  residuals(handle, H.m, N_EIG,H.d_matrix, X.d_matrix, EVa.d_matrix, HX_inter.d_matrix, R.d_matrix);
  
}
  


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
  CHECK_ROCBLAS_STATUS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
  
  // --- rocBLAS Call (Strided Batched NRM2) ---
  std::cout << "Calling rocblas_dnrm2_strided_batched..." << std::endl;
  CHECK_ROCBLAS_STATUS(
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
    return h_norms;
}



/****************
 * STEP 7: Not converged we compute the corrections 
 */

extern void corrections(int M, int N_EIG,
		   double *d_R,      // residuals 
		   double *d_diag_H, // H diagonal
		   double *d_eig_vals, // eigenvalues 
		   double *d_T,         // result
		   const double epsilon);

/***************
 * STEP 8:  V  = V+  Corrections
 */

void corrections_stack(Matrix &R, Matrix &D, Matrix &EVa, Matrix &T, const double epsilon, int N_EIG) {
  
  printf("R "); R.print();
  printf("D "); D.print();
  printf("EVa "); EVa.print();
  printf("T "); T.print();
  corrections(R.m,  N_EIG, R.d_matrix, D.d_matrix, EVa.d_matrix,
	      T.d_matrix + T.m*(T.n), // stack them 
	      epsilon);
  T.n += N_EIG;
}





/***************
 * STEP 9:  V,R = QR(V)
 */

static inline
void QR(rocblas_handle handle,
	Matrix &A,    // the original matrix in the CPU
	Matrix &Tau // we need the pointer 
	) {
  
  CHECK_ROCBLAS_STATUS(rocsolver_dgeqrf(handle, A.m, A.n, A.d_matrix, A.m /*LDA column major*/, Tau.d_matrix));
  CHECK_ROCBLAS_STATUS(rocsolver_dorgqr(handle, A.m, A.n, std::min( A.m, A.n), A.d_matrix, A.m, Tau.d_matrix));
  
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
  bool operator()(const Value_Index& a, const Value_Index& b) const {
    return a.value < b.value; // Use 'a > b' for descending order
  }
};



static inline 
void argsortCpp( std::vector<Value_Index> &A ) {

  std::sort(A.begin(), A.end(), CompareFunction());


}




// Matrices are in column majors 

void davidson_rocm( Matrix  H,    // Hamiltonian matrix ?
		    int num_eigs, // the number of minimum eigen values/vectors 
		    int max_iterations=100,
		    double tolerance=1e-8) {

  printf(" davidson  \n");
  // Create rocBLAS handle
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

  int M = std::min(num_eigs*max_iterations, H.n); // H is square? 

  
  double v_array[H.size()];

  // projections 
  Matrix V = {v_array,H.m, num_eigs};  // projection matrix 
  Matrix T = {0, H.m,V.n};       // H*V
  Matrix TT = {0, H.m,V.n};       // 
  Matrix R  = {0, H.m,V.n};       // 
  Matrix P = {0,V.n,V.n};        // projected  V^t * E

  // Eigensolver
  double eigen_values[num_eigs];
  double eigen_vectors[num_eigs*H.m];

  Matrix EVa = {eigen_values,H.m ,1};    // Eigenvalues
  Matrix Work = {0,H.m ,1};    // Eigenvalues 
  Matrix EVe_sorted = {eigen_vectors,H.m, H.n};  // Sorted 


  
  V.zero();

  // This is what should be 
  
  //  # Initial guess vectors (e.g., identity matrix columns corresponding to lowest diagonal elements)
  //  # Sort diagonal indices to pick the ones corresponding to the lowest expected eigenvalues
  //  diag_indices = np.argsort(np.diag(A))
  //  V = np.zeros((n, num_eigs))
  //  for i in range(num_eigs):
  //      V[diag_indices[i], i] = 1.0


  std::vector<struct Value_Index> diag(H.n);
  double diagonal[H.n];
  for (int i=0;i<H.m; i++) {
    diag[i].value = H.matrix[H.ind(i,i)];
    diagonal[i] = H.matrix[H.ind(i,i)];
    diag[i].index = i;
  }

  Matrix D = {diagonal, H.m,1};
  Matrix Tau = {0, H.m,1};
  
  printf(" Sorting \n");
  argsortCpp(diag);
  
  for (int i=0;i<num_eigs; i++) {
    V.matrix[V.ind(diag[i].index,i)] = 1;
  }


  printf(" Allocating  \n");
  // We copy H into GPU  
  CHECK_HIP_ERROR(hipMalloc(&H.d_matrix, H.size()* sizeof(double)));
  CHECK_HIP_ERROR(hipMemcpy(H.d_matrix , H.matrix, H.size() * sizeof(double), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMalloc(&D.d_matrix, D.size()* sizeof(double)));
  CHECK_HIP_ERROR(hipMemcpy(D.d_matrix , D.matrix, D.size() * sizeof(double), hipMemcpyHostToDevice));

  // This is needed by the QR
  CHECK_HIP_ERROR(hipMalloc(&Tau.d_matrix, Tau.size()* sizeof(double)));
  int *d_perm_indices;
  CHECK_HIP_ERROR(hipMalloc(&d_perm_indices, H.m* sizeof(double)));
  

  // We allocate the largest V and copy the new subspace V into GPU  
  CHECK_HIP_ERROR(hipMalloc(&V.d_matrix,H.m*M* sizeof(double)));
  CHECK_HIP_ERROR(hipMemcpy(V.d_matrix,  V.matrix, V.size() * sizeof(double), hipMemcpyHostToDevice));
  
  
  int *info_device;
  double alpha = 1.0;
  double beta = 0.0;
  double *d_norms;
  const double epsilon = 1e-12;
  
  // We allocate the largest P = V^t * H * V we can afford once
  CHECK_HIP_ERROR(hipMalloc(&d_norms, num_eigs* sizeof(double)));         // Projection -> eigenvectors  
  CHECK_HIP_ERROR(hipMalloc(&P.d_matrix, M*M* sizeof(double)));         // Projection -> eigenvectors  
  CHECK_HIP_ERROR(hipMalloc(&T.d_matrix, H.m*M* sizeof(double)));       // temp
  CHECK_HIP_ERROR(hipMalloc(&TT.d_matrix, H.m*M* sizeof(double)));       // temp
  CHECK_HIP_ERROR(hipMalloc(&R.d_matrix, H.m*M* sizeof(double)));       // temp
  // We allocate the largest T = H*V we can afford once
  

  // 2. Allocate device memory Eigen solver
  CHECK_HIP_ERROR(hipMalloc(&EVa.d_matrix,EVa.size()*sizeof(double)));    // Eigen value
  CHECK_HIP_ERROR(hipMalloc(&EVe_sorted.d_matrix,EVe_sorted.size()*sizeof(double)));    // Eigen vector sorted

  CHECK_HIP_ERROR(hipMalloc(&Work.d_matrix, Work.size()*sizeof(double))); // work space 
  CHECK_HIP_ERROR(hipMalloc(&info_device, sizeof(int)));        // solver burfing 

  printf(" start loop  \n");
  
  for (int iteration=0; iteration<max_iterations; iteration++) {
    R.n = V.n; 
    P.m= P.n =  V.n;
    T.n = V.n;
    EVa.m = V.n;
    EVe_sorted.m = EVe_sorted.n = V.n;

    printf(" Projection  \n");

    // P <- V^t * (T= H * V) projection into a smaller space -> dP is in
    // Python: P = V.T @ (T = H @ V) 
    // the GPU 
    projection(handle, H,V,T,P); 

    printf(" Eigen solver \n");
    
    // dW has the sorted eigenvalue and dP has the sorted eigenvector 
    // Python: eig_vals_T, eig_vecs_T = np.linalg.eigh(T) # rocsolver_dsyev_ 
    eigenSolver(handle,P, EVa, Work,info_device);  
    printf("P "); P.print();
    printf("E "); EVa.print();
    /* Python: 
       idx = np.argsort(eig_vals_T)
       eig_vals_T = eig_vals_T[idx]
       eig_vecs_T = eig_vecs_T[:, idx]
    */

    printf(" Sort arg  \n");
    
    sortarg(P,  EVe_sorted, EVa, d_perm_indices); // dPP is the sorted eigen vectors 
    printf("P "); P.print();
    printf("E "); EVa.print();
    printf("E_S "); EVe_sorted.print();
    EVe_sorted.n =  num_eigs;
    EVa.m =  num_eigs;
    
    
    printf(" Ritz \n");
    printf("V"); V.print();
    printf("E");  EVe_sorted.print();
    printf("T"); T.print();
    
    // V * EigenVectors Called Ritz vectors
    // Python: current_eig_vecs = V @ eig_vecs_T[:, :num_eigs]
    CHECK_ROCBLAS_STATUS(
	   rocblas_dgemm(
	     handle, 
	     rocblas_operation_none, 
	     rocblas_operation_none, 
	     V.m, num_eigs, V.n,  // notice the last dimension
	     &alpha, 
	     V.d_matrix, V.m, // lead V
	     EVe_sorted.d_matrix, EVe_sorted.m, // lead EigenVectors 
	     &beta,   
	     T.d_matrix, T.m  // Ritz 
	   )
     );


    printf(" Residual  \n");
    
    // =======================================================
    // PHASE 1: COMPUTE RESIDUALS d_HX = d_H*d_X
    //          P = H@X -  d_eig_vals*X
    residuals(handle,H,T, EVa,TT,R,num_eigs) ;
    
      

    printf(" Norms  \n");

    std::vector<double> norms = calculate_norm(num_eigs,H.m,
					       R.d_matrix,
					       d_norms,
					       handle);
    int converged_count = 0;
    for ( int i=0; i< num_eigs; i++) 
      if (norms[i] < tolerance)
	converged_count += 1;


    if (converged_count == num_eigs) {
      printf("Davidson converged after %d iteration", iteration);

      CHECK_HIP_ERROR(hipMemcpy(EVa.d_matrix, EVa.matrix ,  num_eigs* sizeof(double), hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(EVe_sorted.d_matrix,EVe_sorted.matrix, EVe_sorted.m*num_eigs* sizeof(double), hipMemcpyDeviceToHost));
      EVa.print(true);
      break;
    }
    printf(" Corrections \n");
    corrections_stack(R,D,EVa,V,epsilon,num_eigs);

    printf(" QR  \n");

    QR(handle,V,Tau);
  }

  CHECK_HIP_ERROR(hipFree(H.d_matrix));
  CHECK_HIP_ERROR(hipFree(D.d_matrix));
  CHECK_HIP_ERROR(hipFree(Tau.d_matrix));
  CHECK_HIP_ERROR(hipFree(d_perm_indices));
  CHECK_HIP_ERROR(hipFree(V.d_matrix));
  CHECK_HIP_ERROR(hipFree(d_norms));
  CHECK_HIP_ERROR(hipFree(P.d_matrix));
  CHECK_HIP_ERROR(hipFree(T.d_matrix));
  CHECK_HIP_ERROR(hipFree(TT.d_matrix));
  CHECK_HIP_ERROR(hipFree(R.d_matrix));
  CHECK_HIP_ERROR(hipFree(EVa.d_matrix));
  CHECK_HIP_ERROR(hipFree(EVe_sorted.d_matrix));
  CHECK_HIP_ERROR(hipFree(Work.d_matrix));
  CHECK_HIP_ERROR(hipFree(info_device));

  
}
    
  

int main(int argc, char* argv[]) {
  
  // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = std::atoi(argv[1]);
  std::cout << "Integer from atoi: " << mydevice << std::endl;
  int result = set_device(mydevice);
  int M = std::atoi(argv[2]);;
  int n_eng =std::atoi(argv[3]);

  
  double hH[M * M];    
  Matrix H = {hH, M,M};

  H.zero();
  for (int i = 0; i<M; i++) {
    H.matrix[H.ind(i,i)] = i+1;
  }
  
    
  davidson_rocm(H,
		n_eng,
		100,
		1e-8);

  
}


