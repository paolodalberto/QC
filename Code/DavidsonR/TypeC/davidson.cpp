/**********************************
 **  This is a code by Gemini and I 
 ** 
 ***/




#include "davidson.h"




void GEMMP(Matrix &C, ZC beta, Matrix &A, Matrix &B, ZC alpha, rocblas_handle handle=0) {

  // T = H * V
  CHECK_ROCBLAS_STATUS(
	GEMM(handle, 
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
		ZC alpha = ALPHA,  ZC beta = BETA
) {


  //printf("H \n"); H.print();
  //printf("V \n"); V.print();
  //printf("T \n"); T.print();
  // T = H * V
  CHECK_ROCBLAS_STATUS(
	GEMM(handle, 
	     rocblas_operation_none, rocblas_operation_none, 
	     H.m, V.n, H.n,  // this is the problem size                                                 
	     &alpha, 
	     H.d_matrix, H.m, 
	     V.d_matrix, V.m, 
	     &beta, 
	     T.d_matrix, T.m)); 

  //printf("P \n"); P.print();
  // P = V^t T 
  CHECK_ROCBLAS_STATUS(
	GEMM(handle, 
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
	SOLV(
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
		    ZC *d_A, // source matrix  N rows and M columns we swap the columns
		    ZC *d_B, // permuted destination matrix 
		    ZC *d_sort_keys, // sorting keys
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
		      ZC *d_H, 
		      ZC *d_X, // Eigenvectors 
		      ZC *d_eig_vals, // eigenvalues 
		      ZC *d_HX_inter, // intermediate result
		      ZC *d_R  // residuals
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

std::vector<NORM_TYPE>
calculate_norm(
	       int N, 
	       int M,
	       ZC* d_A,
	       NORM_TYPE* d_norms,
	       rocblas_handle handle) {
  
  std::vector<NORM_TYPE> h_norms(N);
  
  // Set pointer mode to device, as results are stored on the device
  //  CHECK_ROCBLAS_STATUS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
  
  // --- rocBLAS Call (Strided Batched NRM2) ---
  //std::cout << "Calling rocblas_dnrm2_strided_batched..." << std::endl;
  CHECK_ROCBLAS_STATUS(
	 NORM(
	      handle,
	      M,               // n: length of each vector
	      d_A,             // x: device pointer to the matrix/vectors
	      1,               // incx: stride within each vector (1 for column-major matrix)
	      M,               // stride_x: stride between consecutive vectors in memory
	      N,               // batch_count: number of vectors (columns)
	      d_norms          // result: device pointer to store norms
	      ));

    // --- Copy results back to host ---
    CHECK_HIP_ERROR(hipMemcpy(h_norms.data(), d_norms, N * sizeof(NORM_TYPE), hipMemcpyDeviceToHost));
    return h_norms;
}



/****************
 * STEP 7: Not converged we compute the corrections 
 */

extern void corrections(int M, int N_EIG,
		   ZC *d_R,      // residuals 
		   ZC *d_diag_H, // H diagonal
		   ZC *d_eig_vals, // eigenvalues 
		   ZC *d_T,         // result
		   const ZC epsilon);

/***************
 * STEP 8:  V  = V+  Corrections
 */

void corrections_stack(Matrix &R, Matrix &D, Matrix &EVa, Matrix &T, const ZC epsilon, int N_EIG) {
  
  //printf("R "); R.print();
  //printf("D "); D.print();
  //printf("EVa "); EVa.print();
  //printf("T "); T.print();
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
  //printf("A "); A.print();
  //printf("Tau "); Tau.print();
  
  CHECK_ROCBLAS_STATUS(GEQRF(handle, A.m, A.n, A.d_matrix, A.m /*LDA column major*/, Tau.d_matrix));
  CHECK_ROCBLAS_STATUS(ORGQR(handle, A.m, A.n, std::min( A.m, A.n), A.d_matrix, A.m, Tau.d_matrix));
  
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

  int debug = 0;
  if (true || debug) printf(" davidson  %d %d \n",num_eigs,max_iterations);
  // Create rocBLAS handle
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

  int M = std::min(num_eigs*max_iterations, H.n); // H is square? 

  // projections 
  Matrix V  = {H.m, num_eigs, H.m, M};  // projection matrix 
  Matrix T  = {H.m, V.n, H.m, M     };  // H*V
  Matrix TT = {H.m, V.n, H.m, M};       // 
  Matrix R  = {H.m, V.n, H.m, M};       // 
  Matrix P  = {V.n, V.n, M,M};        // projected  V^t * E


  V.alloc(true,true);  // Starting V 
  P.alloc(false,true);  // Projection V^T H V 
  T.alloc(false,true);  // temp -- H*V 
  TT.alloc(false,true); // Another temp 
  R.alloc(false,true);  // Residuals  

  // Eigensolver
  Matrix EVa  = {H.n ,1,M ,1 };    // Eigenvalues
  Matrix Work = {P.m ,P.n,M ,M };  // Eigenvalues 
  Matrix EVe_sorted = {P.m, P.n, M, M};  // Sorted 

  EVa.alloc(true,true);
  Work.alloc(false,true);
  EVe_sorted.alloc(true,true);
  
  // This is what should be 
  
  //  # Initial guess vectors (e.g., identity matrix columns corresponding to lowest diagonal elements)
  //  # Sort diagonal indices to pick the ones corresponding to the lowest expected eigenvalues
  //  diag_indices = np.argsort(np.diag(A))
  //  V = np.zeros((n, num_eigs))
  //  for i in range(num_eigs):
  //      V[diag_indices[i], i] = 1.0


  std::vector<struct Value_Index> diag(H.n);

  Matrix D   = {H.m,1,H.m,1};
  Matrix Tau = {M,1,M,1 };

  D.alloc(true,true);
  Tau.alloc(false,true); // required for QR

  ZC diagonal[H.n];
  for (int i=0;i<H.m; i++) {
    diag[i].value = H.matrix[H.ind(i,i)];
    D.matrix[i]   = H.matrix[H.ind(i,i)];
    diag[i].index = i;
  }

  
  if (debug) printf(" Sorting \n");
  argsortCpp(diag);
  
  for (int i=0;i<num_eigs; i++) {
    V.matrix[V.ind(diag[i].index,i)] = 1;
  }
  
  if (debug) printf(" Move D \n");
  D.writetodevice();
  if (debug) printf(" Move H \n");
  H.writetodevice();

  if (debug) printf(" Move D and H \n");
  ZC alpha = ALPHA;
  ZC beta = BETA;
  int *d_perm_indices;
  int *info_device;
  NORM_TYPE *d_norms;
  const ZC epsilon = 1e-12;
  
  CHECK_HIP_ERROR(hipMalloc(&d_perm_indices, H.m* sizeof(int))); 
  CHECK_HIP_ERROR(hipMalloc(&d_norms, num_eigs* sizeof(NORM_TYPE)));         
  CHECK_HIP_ERROR(hipMalloc(&info_device, sizeof(int)));        // solver burfing 

  if (debug) printf(" start loop  \n");
  V.writetodevice();  
  for (int iteration=0; iteration<max_iterations; iteration++) {
    //V.readfromdevice(); printf("V "); V.print(true);
    //V.writetodevice();  


    R.n = V.n; 
    P.m= P.n =  V.n;
    T.n = V.n;
    EVa.m = V.n;
    EVe_sorted.m = EVe_sorted.n = V.n;

    if (debug) printf(" Projection  \n");
    // P <- V^t * (T= H * V) projection into a smaller space -> dP is in
    // Python: P = V.T @ (T = H @ V) 
    // the GPU 
    projection(handle, H,V,T,P); 
    
    //P.readfromdevice(); P.print(true);
    if (debug) printf(" Eigen solver \n");
    
    // dW has the sorted eigenvalue and dP has the sorted eigenvector 
    // Python: eig_vals_T, eig_vecs_T = np.linalg.eigh(T) # rocsolver_dsyev_ 
    eigenSolver(handle,P, EVa, Work,info_device);  

    /* Python: 
       idx = np.argsort(eig_vals_T)
       eig_vals_T = eig_vals_T[idx]
       eig_vecs_T = eig_vecs_T[:, idx]
    */

    if (debug) printf(" Sort arg  \n");
    
    sortarg(P,  EVe_sorted, EVa, d_perm_indices); // dPP is the sorted eigen vectors 
    
    EVe_sorted.n =  num_eigs;
    EVa.m =  num_eigs;

    //EVa.readfromdevice();
    //EVe_sorted.readfromdevice();
    //EVa.print(true);
    //EVe_sorted.print(true);

    
    
    if (debug) printf(" Ritz \n");
    
    // V * EigenVectors Called Ritz vectors
    // Python: current_eig_vecs = V @ eig_vecs_T[:, :num_eigs]
    CHECK_ROCBLAS_STATUS(
	   GEMM(
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
    
    
    //T.readfromdevice(); printf("After T"); T.print(true);

    if (debug) printf(" Residual  \n");
    
    
    // =======================================================
    // PHASE 1: COMPUTE RESIDUALS d_HX = d_H*d_X
    //          P = H@X -  d_eig_vals*X

    residuals(handle,H,T, EVa,TT,R,num_eigs) ;
    
    if (debug) printf(" Norms  \n");

    std::vector<NORM_TYPE> norms = calculate_norm(num_eigs,H.m,
					       R.d_matrix,
					       d_norms,
					       handle);


    projection(handle, H,V,T,P); 
    int converged_count = 0;
    //printf(" Iteration %d \n",iteration); 
    for ( int i=0; i< num_eigs; i++) { 
      if (i ==0) printf("Norm[%d]= %e vs %e \n", i,norms[i], tolerance);
      if (norms[i] < tolerance) { 
	converged_count += 1;
      }
    }

    if (converged_count == num_eigs) {
      printf("Davidson converged after %d iteration\n", iteration);

      EVa.readfromdevice();
      EVe_sorted.readfromdevice();
      for (int i=0; i<num_eigs; i++)
	printf(" %d %e \n",i,EVa.matrix[i]);
      //EVe_sorted.print(true);
      break;
    }
    
    if (debug) printf(" Corrections \n");
    corrections_stack(R,D,EVa,V,epsilon,num_eigs);
    
    if (debug) printf(" QR  \n");
    
    QR(handle,V,Tau); // V is input and output ,
    // Tau is used as temporary space ...

  }



  CHECK_HIP_ERROR(hipFree(d_perm_indices));
  CHECK_HIP_ERROR(hipFree(d_norms));
  CHECK_HIP_ERROR(hipFree(info_device));

  Tau.free();
  D.free();
  EVe_sorted.free();
  Work.free();
  EVa.free();
  R.free();
  TT.free();
  T.free();
  P.free();
  V.free();

  CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));
  
}
    
void normal(Matrix H,int n_eng) {
  int *info_device;
  rocblas_handle handle;
  int *d_perm_indices;

  H.writetodevice();
  CHECK_HIP_ERROR(hipMalloc(&d_perm_indices, H.m* sizeof(double)));
  CHECK_HIP_ERROR(hipMalloc(&info_device, sizeof(int)));        // solver burfing 
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

  Matrix EVa  = {H.n ,1,H.n ,1 };    // Eigenvalues
  Matrix Work = {H.m ,H.n, H.m,H.m };  // Eigenvalues 
  Matrix EVe_sorted = {H.m, H.n,H.m, H.n,};  // Sorted 

  
  EVa.alloc(true,true);
  Work.alloc(false,true);
  EVe_sorted.alloc(true,true);
  
  eigenSolver(handle,H, EVa, Work,info_device); 
  sortarg(H,  EVe_sorted, EVa, d_perm_indices); // 

  EVa.readfromdevice();
  EVe_sorted.readfromdevice();
  printf(" Normal \n");
  for (int i=0; i<n_eng; i++)
    printf(" %d %e \n",i,EVa.matrix[i]);


  EVa.free();
  Work.free();
  EVe_sorted.free();
  CHECK_HIP_ERROR(hipFree(d_perm_indices));
  CHECK_HIP_ERROR(hipFree(info_device));
  
  
}


#include <iostream>
#include <chrono>
#include <thread> // For std::this_thread::sleep_for

void longRunningFunction() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
}




int main(int argc, char* argv[]) {
  
  // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = (argc>1)? std::atoi(argv[1]):0;
  std::cout << "Integer from atoi: " << mydevice << std::endl;
  int result =  set_device(mydevice);
  int M = (argc>2)? std::atoi(argv[2]):1000;
  int n_eng =  (argc>3)?std::atoi(argv[3]):1;
  int it    = (argc>4)?std::atoi(argv[4]):3;  
  
  printf(" device: %d M: %d n_eng: %d it: %d \n", mydevice, M, n_eng, it); 
  printf("SIZE = %lu \n", sizeof(ZC));

  Matrix H = {M,M,M,M};
  H.alloc(true,true);

  H.zero();
  for (int i = 0; i<M; i++) {
    H.matrix[H.ind(i,i)] = i+1;
    //printf(" i %d index %d M %f \n", i, H.ind(i,i), H.matrix[H.ind(i,i)]);
  }
  for (int i = 0; i<M; i++) {
    for (int j = i+1; j<M; j++) {
      H.matrix[H.ind(i,j)] = 1.0/(i+j+2);
      H.matrix[H.ind(j,i)] = 1.0/(i+j+2);
    }
  }

  //H.print(true);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  davidson_rocm(H,
		n_eng,
		std::min(it, H.m/n_eng),
#if(TYPE_OPERAND==0  )
		1e-3
#elif(TYPE_OPERAND==1 )
		1e-4
#else
		1e-8
#endif
	       
		);

  // Record end time
  auto end = std::chrono::high_resolution_clock::now();
  
  // Calculate duration in seconds
  std::chrono::duration<double> elapsed_seconds = end - start;
  
  // Output the elapsed time
  std::cout << "davidson: " << elapsed_seconds.count() << " seconds.\n";
  if (0) {
    start = std::chrono::high_resolution_clock::now();
    normal(H,n_eng);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "normal: " << elapsed_seconds.count() << " seconds.\n";
  }
  H.free();
  
  
}


