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



void gate::init(Matrix I, Matrix O) {
  
  
  
  // already allocated nothing to
  U.alloc(true, true);
  
  // remember we are doing O = I* U (where U is U^t)
  
  int B = I.m; // this is the state 2^n 
  batch_count = B - ((1<<bit_number)*((bit_number==0)?0:1)
		     +U.m);
  
  batch_count = std::max(1, batch_count);
  printf("init\n");
  U.print(true);
  
  
  m = U.m;
  n = 1<<bit_number;;
  k = U.n; 
  printf(" m %d n %d k %d \n", m, n, k);
  printf(" B %d batch_count %d bit_number %d \n", B, batch_count, bit_number);
  
  if (0) { 
    alloc(true,true);
    pre_gpu_gemm(m,n,k,
		 h_A_ptrs,U.m,  U.d_matrix,
		 h_B_ptrs,m,    I.d_matrix,
		 h_C_ptrs,m,    O.d_matrix,
		 batch_count);
    
    CHECK_HIP_ERROR(hipMemcpy(d_A_ptrs, h_A_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B_ptrs, h_B_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C_ptrs, h_C_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    
  }
}





void gate::step(rocblas_handle handle,
		Matrix &I, Matrix &O,
		int count ) {
  
  Matrix Z{4,1,4,1};
  Z.alloc(true,false);
  printf("######   Step %d \n", count );
  U.print();
  printf(" m %d n %d k %d bc %d \n", m, n, k,batch_count);
  
  const rocblas_stride strideA = m*k;
  const rocblas_stride strideB = k*n;
  const rocblas_stride strideC = m*n;
  
  printf(" stridaA %ld strideB %ld strideC %ld \n",
	 strideA,strideB, strideC);
  
  U.print(true);
  I.print(true);
  O.print(true);
  
  cpu_zgemm_batched(
		    m, n, k, alpha, 
		    U.matrix, m,
		    I.matrix, m,
		    beta,
		    O.matrix, m,
		    batch_count);

  printf("Z cpu \n");
  O.print(true);
  
  if (0) 
    gpu_zgemm_batched(
		      handle,
		      m, n, k, alpha, 
		      d_A_ptrs, strideA,
		      d_B_ptrs, strideB,
		      beta,
		      d_C_ptrs, strideC,
		      batch_count
		      );
  
  printf("######   Step %d \n \n",count);
  
}




  // we move all the matrices into the 
void schedule::init(){
    printf("Circuit init \n");
    for (std::vector<Gate> &level  : schedule)
      for (Gate &h : level ) { 
	h.init(I,O);
      }
    printf("Circuit init \n\n");
  }

  
void schedule::forward(rocblas_handle handle) {
  int count = 0;
  printf("Circuit forward \n");
  
  int lvl =0 ;
  for (std::vector<Gate> &level  : schedule)
    
    for (Gate &h : level ) { 
      
      I.print(true);
      //I.writetodevice();
      h.step(handle,
	     (lvl%2==0)?I:O,
	     (lvl%2==0)?O:I,
	     count++);
      lvl ++;
      //O.readfromdevice();
      O.print(true);
    }
  printf("Circuit forward \n\n");
}



void schedule::print(bool  t)  {
    int l =0;
    printf("BEGIN Circuit %zu \n", schedule.size());
    I.print(true);
    for (std::vector<Gate> &level  : schedule) {
      printf("Level %d < %zu \n", l++, level.size());
      for (Gate &h : level ) 
	h.print(t);
    }
    printf("END Circuit \n");
  }










// Construction of gates
static NORM_TYPE SQ2 = 1.0/std::sqrt(2);
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
  ZERO, ZERO, ZERO, ONE,
  ZERO, ONE,  ZERO, ZERO,
  ZERO, ZERO, ONE,  ZERO}; 
Matrix CNM_{4,4,4,4, cnot_matrix};
Gate CNot{"CNot", CNM_};

	     


