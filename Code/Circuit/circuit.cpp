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


// Construction of gates
static NORM_TYPE SQ2 = std::sqrt(2);
/** 1/sqrt(2) | 1  1 |
 *            | 1 -1 | 
 * Stored in column major 
 */

static ZC hadamard_matrix[] =  {  SQ2*ONE, SQ2*ONE, SQ2*ONE,-SQ2*ONE }; 
Matrix HM_{2,2,2,2, hadamard_matrix};
Gate Hadamard{"hadamard", HM_};

/**  | 1  0 |
 *   | 0  1 | 
 * Stored in column major but complex numbers  
 */
static ZC identity_matrix[] =  {  ONE, ZERO, ZERO, ONE }; 
Matrix IM_{2,2,2,2, identity_matrix};
Gate Identity{"Identity",IM_};

/**  | 0 1 |
 *   | 1 0 | 
 * Stored in column major but complex numbers  x() or not
 */
static ZC pauli_x_matrix[] =  {  ZERO, ONE, ONE, ZERO };
Matrix PXM_{2,2,2,2, pauli_x_matrix};
Gate Pauli_X{"PauliX", PXM_};

/**  | 0  -i |
 *   | i  0 | 
 * Stored in column major but complex numbers  
 */

static ZC pauli_y_matrix[] =  { ZERO, 1.0i*ONE, -1.0i* ONE, ZERO  }; 
Matrix PYM_{2,2,2,2, pauli_y_matrix};
Gate Pauli_Y{"PauliY",PYM_};

/**  | 1  0 |
 *   | 0  -1 | 
 * Stored in column major but complex numbers  
 */
static const ZC pauli_z_matrix[] =  { ONE, ZERO, ZERO, -ONE }; 
Matrix PZM_{2,2,2,2, pauli_x_matrix};
const Gate Pauli_Z{"PauliZ", PZM_};



/**  | 1 0 0 0 |
 *   | 0 1 0 0 |
 *   | 0 0 0 1 |
 *   | 0 0 1 0 |
 
 * Stored in column major but complex numbers  
 */
static  ZC cnot_matrix[] =  {
  ONE,  ZERO, ZERO, ZERO,
  ZERO, ONE,  ZERO, ZERO,
  ZERO, ZERO, ZERO, ONE,
  ZERO, ZERO, ONE,  ZERO}; 
Matrix CNM_{4,4,4,4, cnot_matrix};
Gate CNot{"CNot", CNM_};

	     


