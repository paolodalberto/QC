
#pragma once

#define TYPE_OPERAND 4

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
static ZC EPS{ 1e-6, 1e-6};
#define GEMM  rocblas_cgemm
#define SOLV  rocsolver_cheev
#define NORM_  rocblas_scnrm2_strided_batched
#define GEQRF rocsolver_cgeqrf 
#define ORGQR  rocsolver_cungqr
#endif


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
template <typename Entry>
struct matrix {

  // struct as a class members are all public
  int m;                       // rows
  int n;                       // cols
  int M;                       // Maximum rows LD
  int N;                       // Maximum cols
  Entry *matrix = 0; // host  
  Entry *d_matrix =0;            // device

  void free() {
    if (matrix) std::free(matrix);
    if (d_matrix) CHECK_HIP_ERROR(hipFree(d_matrix));
  }
  void alloc(bool host , bool device  ) {
    if (size()>0) {
      //printf(" Allocated %d * %d = %d elements \n", M, N,M*N);
      if (host)   matrix = (Entry*) std::calloc(M*N,sizeof(Entry));
      if (device) CHECK_HIP_ERROR(hipMalloc(&d_matrix, M*N* sizeof(Entry)));		      
    }
  }
  void readfromdevice() {
         CHECK_HIP_ERROR(hipMemcpy(matrix , d_matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));
  }
  void writetodevice() {
         CHECK_HIP_ERROR(hipMemcpy(d_matrix , matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));

  }
  // functions 
  // Host-side matrix multiplication for verification (optional)
  void (*gemm)(struct matrix &C, Entry beta, struct matrix &A, struct matrix &B, Entry alpha);
  void init() {
    for (int i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(i % 10);
  };
  void zero() {
    for (int i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(0);
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
	    //printf("%0.2f %d %d %d ", matrix[ind(i,j)],i,j,ind(i,j));
	    printf("%e ", matrix[ind(i,j)]);
	}
	printf("\n");
      }
  };
};


typedef struct matrix<ZC> Matrix;
typedef struct matrix<NORM_TYPE> EigenValueMatrix;
