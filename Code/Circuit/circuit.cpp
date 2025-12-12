/*********************************************************
 * The Idea is simple: The computation of a Gate G on a state S is
 * expressed by a a kronecker computation (x) as 
 *
 * (I_n (x) G (x) I_k) * S
 *
 * I_n stands for an identity matrix and n amd k mean the number of
 * bit that is the Gate applied to m bits the G is applied to bit k,
 * k+1, ... k+m-1 and the total number of bits is n + m + k and the
 * state is of size 2^(n+m+k). 
 *
 * We showed this in the Python implementation and also we show that
 * this boils down to the computation of 2^n matrix multiplicaitons
 * (strided) G over the remainder state S[0... 2^(k+m)] as a matrix of
 * size 2^m x 2^k ... so far brilliant
 *
 **************/


#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>



// we define the computation double complex 

#define TYPE_OPERAND 4 
#include "davidson.h"  // definition of matrices 
#include "circuit.h"   // definition of Gate and Circuit

static SQ2 = std::sqrt(x);

static const ZC hadamard_matrix[] =  {  SQ2*ONE, SQ2*ONE, SQ2*ONE,-SQ2*ONE }; 
const Gate Hadamard("hadamard",
	      Matrix H{2,2,2,2, hadamard_matrix}
	      );

static const ZC identity_matrix[] =  {  ONE, ZERO, ZERO, ONE }; 
const Gate Identity("Identity",
	      Matrix H{2,2,2,2,identity_matrix}
	      );

static const ZC pauli_x_matrix[] =  {  ZERO, ONE, ONE, ZERO }; 
const Gate Pauli_X("Identity",
	      Matrix H{2,2,2,2,identity_matrix}
	      );

static const ZC pauli_y_matrix[] =  { ZERO, 1.0i, -1.0i* ONE, ZERO  }; 
const Gate Pauli_Y("Identity",
	      Matrix H{2,2,2,2,identity_matrix}
	      );
static const ZC pauli_y_matrix[] =  { ONE, ZERO, ZERO, -ONE }; 
const Gate Pauli_Z("Identity",
	      Matrix H{2,2,2,2,identity_matrix}
	      );

  
	     
