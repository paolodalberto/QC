
#pragma once

/**
 *  TYPE_OPERAND 0 HALF precision ... not working
 *  TYPE_OPERAND 1 float
 *  TYPE_OPERAND 2 double 
 *  TYPE_OPERAND 3 float complex 
 *  TYPE_OPERAND 0 double complex
 */ 

#ifndef TYPE_OPERAND 
#define TYPE_OPERAND 4
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
//#include <rocsolver/rocsolver.h>
#include <cstdlib>
#include <cblas.h>
#include <map>


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
#define GEAM  rocblas_sgeam
#define GEMMC  cblas_sgemm
#define SOLV  rocsolver_ssyev
#define NORM_  rocblas_snrm2_strided_batched
#define GEQRF rocsolver_sgeqrf 
#define ORGQR rocsolver_sorgqr 
static ZC EPS  = 1e-10;
#define GEMM_BATCHED rocblas_sgemm_batched

#elif (TYPE_OPERAND==2)
typedef double NORM_TYPE;
typedef double ZC;
static ZC ALPHA = 1.0;
static ZC BETA  = 0.0;
static ZC ONE = 1.0;
static ZC ZERO  = 0.0;
#define GEMM  rocblas_dgemm
#define GEMMC  cblas_dgemm
#define GEAM  rocblas_dgeam
#define SOLV  rocsolver_dsyev
#define NORM_  rocblas_dnrm2_strided_batched
#define GEQRF rocsolver_dgeqrf 
#define ORGQR rocsolver_dorgqr 
static ZC EPS  = 1e-12;
#define GEMM_BATCHED rocblas_dgemm_batched


#elif  (TYPE_OPERAND==4)
typedef double NORM_TYPE;
typedef  rocblas_double_complex ZC;
static ZC ALPHA{1.0,0.0};
static ZC BETA{0.0,0.0};
static ZC ONE{1.0,0.0};
static ZC ZERO{0.0,0.0};
static ZC EPS{ 1e-12, 1e-12};

#define GEMM  rocblas_zgemm
#define GEMMC  cblas_zgemm

#define GEAM  rocblas_zgeam

#define SOLV  rocsolver_zheev
#define NORM_ rocblas_dznrm2_strided_batched
#define GEQRF rocsolver_zgeqrf 
#define ORGQR rocsolver_zungqr
#define GEMM_BATCHED rocblas_zgemm_batched

#elif  (TYPE_OPERAND==3)
typedef float NORM_TYPE;
typedef rocblas_float_complex ZC;
static ZC ALPHA{1.0,0.0};
static ZC BETA{0.0,0.0};
static ZC ONE{1.0,0.0};
static ZC ZERO{0.0,0.0};
static ZC EPS{ 1e-6, 1e-6};
#define GEMM  rocblas_cgemm
#define GEMMC  cblas_cgemm

#define GEAM  rocblas_cgeam
#define SOLV  rocsolver_cheev
#define NORM_  rocblas_scnrm2_strided_batched
#define GEQRF rocsolver_cgeqrf 
#define ORGQR  rocsolver_cungqr
#define GEMM_BATCHED rocblas_cgemm_batched
#endif

// Index computations are are as important as the computation type, we
// we just hide the type but also we make sure that when we change
// system we can address more than 2G elements ... we are aiming to
// have a state of size < 16GB .. 2^32 

typedef size_t Index;


#define CHECK_HIP_ERROR(call) {						\
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

