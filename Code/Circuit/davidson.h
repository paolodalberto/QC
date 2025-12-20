
#pragma once

/**
 *  TYPE_OPERAND 0 HALF precision ... not working
 *  TYPE_OPERAND 1 float
 *  TYPE_OPERAND 2 double 
 *  TYPE_OPERAND 3 float complex 
 *  TYPE_OPERAND 0 double complex
 */ 

#ifndef TYPE_OPERAND 
#define TYPE_OPERAND 3
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cstdlib> 

/// Miscellaneous errors

#if (TYPE_OPERAND==0)
typedef float NORM_TYPE;
typedef std::float_t ZC;
static ZC ALPHA = 1.0;
static ZC BETA  = 0.0;
static ZC ONE = 1.0;
static ZC ZERO  = 0.0;
static ZC EPS  = 1e-6;
#define GEMM  rocblas_hgemm
#define SOLV  rocsolver_hsyev
#define NORM_  rocblas_hnrm2_strided_batched
#define GEQRF rocsolver_hgeqrf 
#define ORGQR rocsolver_horgqr 

#elif (TYPE_OPERAND==1)
typedef float NORM_TYPE;
typedef float ZC;
static ZC ALPHA = 1.0;
static ZC BETA  = 0.0;
static ZC ONE = 1.0;
static ZC ZERO  = 0.0;
#define GEMM  rocblas_sgemm
#define SOLV  rocsolver_ssyev
#define NORM_  rocblas_snrm2_strided_batched
#define GEQRF rocsolver_sgeqrf 
#define ORGQR rocsolver_sorgqr 
static ZC EPS  = 1e-10;

#elif (TYPE_OPERAND==2)
typedef double NORM_TYPE;
typedef double ZC;
static ZC ALPHA = 1.0;
static ZC BETA  = 0.0;
static ZC ONE = 1.0;
static ZC ZERO  = 0.0;
#define GEMM  rocblas_dgemm
#define SOLV  rocsolver_dsyev
#define NORM_  rocblas_dnrm2_strided_batched
#define GEQRF rocsolver_dgeqrf 
#define ORGQR rocsolver_dorgqr 
static ZC EPS  = 1e-12;

#elif  (TYPE_OPERAND==4)
typedef double NORM_TYPE;
typedef  rocblas_double_complex ZC;
static ZC ALPHA{1.0,0.0};
static ZC BETA{0.0,0.0};
static ZC ONE{1.0,0.0};
static ZC ZERO{0.0,0.0};
static ZC EPS{ 1e-12, 1e-12};

#define GEMM  rocblas_zgemm
#define SOLV  rocsolver_zheev
#define NORM_ rocblas_dznrm2_strided_batched
#define GEQRF rocsolver_zgeqrf 
#define ORGQR rocsolver_zungqr

#elif  (TYPE_OPERAND==3)
typedef float NORM_TYPE;
typedef rocblas_float_complex ZC;
static ZC ALPHA{1.0,0.0};
static ZC BETA{0.0,0.0};
static ZC ONE{1.0,0.0};
static ZC ZERO{0.0,0.0};
static ZC EPS{ 1e-6, 1e-6};
#define GEMM  rocblas_cgemm
#define SOLV  rocsolver_cheev
#define NORM_  rocblas_scnrm2_strided_batched
#define GEQRF rocsolver_cgeqrf 
#define ORGQR  rocsolver_cungqr
#endif

// Index computations are are as important as the computation type, we
// we just hide the type but also we make sure that when we change
// system we can address more than 2G elements ... we are aiming to
// have a state of size < 16GB .. 2^32 

typedef int Index;


#define CHECK_HIP_ERROR(call) {			\
    hipError_t err = call;						\
    if (err != hipSuccess) {						\
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " : " \
		<< hipGetErrorString(err) << std::endl;			\
      exit(EXIT_FAILURE);						\
    }									\
  } 

#define CHECK_ROCBLAS_STATUS(call)  {					\
    rocblas_status status = call;					\
    if (status != rocblas_status_success) {				\
      std::cerr << "rocm error at " << __FILE__ << ":" << __LINE__ << ":" \
		<< rocblas_status_to_string(status) << std::endl;	\
      exit(EXIT_FAILURE);						\
    }									\
  }

// I like to wrap the matrices into a container
template <typename Entry>
struct matrix {

  // struct as a class members are all public
  Index m;                       // rows
  Index n;                       // cols
  Index M;                       // Maximum rows LD
  Index N;                       // Maximum cols
  Entry *matrix = 0; // host  
  Entry *d_matrix =0;            // device
  bool gate = true;
  bool transpose= false; 
  size_t initialize_host_matrix(const Entry* initial_data) {
    // Calculate total size needed
    size_t total_elements = m * n;
    size_t total_bytes = total_elements * sizeof(Entry);

    // 1. Allocate the memory using malloc
    alloc(true,false);

    if (matrix == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
	return 0;
    }

    memcpy(matrix, initial_data, total_bytes);
    return total_bytes;
  }
  void free() {
    if (matrix and !gate)   { std::free(matrix);        matrix =0; }
    if (d_matrix) { CHECK_HIP_ERROR(hipFree(d_matrix)); d_matrix=0;}
  }
  void alloc(bool host , bool device  ) {
    if (size()>0) {
      //printf(" Allocated %d * %d = %d elements \n", M, N,M*N);
      if (host and matrix==0)   {
	matrix = (Entry*) std::calloc(M*N,sizeof(Entry));
	assert(matrix!=0 && " Failed to allocate Doh\n");
      }
      if (device and d_matrix==0) CHECK_HIP_ERROR(hipMalloc(&d_matrix, M*N* sizeof(Entry)));		      
    }
  }
  void readfromdevice() {
    if ( matrix!=0 and d_matrix!=0) 
      CHECK_HIP_ERROR(hipMemcpy(matrix , d_matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));
  }
  void writetodevice() {
    if ( matrix!=0 and d_matrix!=0) 
      CHECK_HIP_ERROR(hipMemcpy(d_matrix , matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));

  }
  // functions 
  // Host-side matrix multiplication for verification (optional)
  void gemm(struct matrix &C, Entry beta, struct matrix &A,
	    struct matrix &B, Entry alpha, const int debug1=0) {
    for (Index m = 0; m < C.m; ++m) 
      for (Index n = 0; n < C.n; ++n) {
	ZC sum = ZERO;

	for (Index k = 0; k < A.n; ++k) { 
	  sum = sum +A.matrix[A.ind(m,k)]*B.matrix[B.ind(k,n)];
	  if (debug1) std::cout << A.matrix[A.ind(m,k)] << " * " << B.matrix[B.ind(k,n)]<<" = " << sum << "\n";
	}
	C.matrix[C.ind(m,n)] = alpha*sum +  C.matrix[C.ind(m,n)]*beta;
	if (debug1) std::cout <<  " indx " << C.ind(m,n)  << "<- " <<  sum << "\n";
      }
  }
  void init() {
    for (Index i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(i % 10);
  };
  void zero() {
    for (Index i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(0);
  };
  void bra_zero() {
    zero();
    matrix[0] = ALPHA;
  };
  Index ind(Index i, Index j, bool t=false)    {
    if (t || transpose )
      return i*n +j;
    else
      return i +m*j;
    
  }
  
  size_t size() { return m*n; } 
  void print(bool t=false) {
    Index MM = (m>10)?10: m;
    Index NN = (n>10)?10: n;
    std::cout << "Column Major M,N = "<< m << "," << n << "\n";
    if (t)
      for (Index i = 0; i < MM; ++i) {
	for (Index j = 0; j < NN; ++j) 
	  std::cout << matrix[ind(i,j)] << " " ;
	printf("\n");
      }
  };
};


typedef struct matrix<ZC> Matrix;
typedef struct matrix<NORM_TYPE> EigenValueMatrix;
