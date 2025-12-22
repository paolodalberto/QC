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
#include "matrices.h"  // definition of matrices 
#include "circuit.h"   // definition of Gate and Circuit


static int debug1= 0;



/*
 * Helper function for host-side batched strided matrix multiplication
 * verification (standard C++ implementation), the layout is column
 * major this is literally this computation (I_n (x) G (x) I_k) * S
 */
void cpu_zgemm_gate(
     Index M, Index N, Index K,  
     ZC* AR,            // square single matrix MxK stored in column major
     ZC* BR,            // vector  LB elements
     ZC* CR, Index LB,    // vector  LC = LB elements 
     Index batch) {
  
  ZC *A = AR;   // there is only one 
  ZC *B;        // we will stream B and C
  ZC *C;

  // We split S -> B and C  into batch parts
  Index chunk = LB/batch; 
  Index LD    = chunk/M; // logical but very convenient
  
  if (N==1 or batch==1) LD=1;

  if (debug1) std::cout <<  "Chunk "<<  chunk << "LD " << LD << "\n";
  
  for (Index p = 0; p < batch; ++p) {
    /* these are independent GEMM */

    B = BR+ p*chunk; // chunk = MxLD
    C = CR+ p*chunk;
    
    //  C = U*B but B = KxN should be stored row major but it is no
    
    for (Index m = 0; m < M; ++m) {
      for (Index n = 0; n < N; ++n) {
	ZC sum = ZERO;
	for (Index k = 0; k < K; ++k){ 

	  // we go across rows for A in the inner loop and across B
	  // technically correct but inefficient and this is not
	  // somthing you can do easily calling BLAS
	  
	  sum = sum +A[m +k*M]*B[k*LD+n];

	  if (debug1) std::cout <<  A[m +k*M]<< " * " << B[k*N+n]<<" = " << sum << "\n";
	}
	C[m*N+n] = sum;
	if (debug1) std::cout <<  " indx " <<  m*N+n << "<- " <<  sum << "\n";
      }
    }
    
  }
}

/*
 * This is as above A * B but we transpose and we actually compute the
 * B^t * A^t : this is cleaner when we use column major and A = A^t
 * (but we may just store the transpose or transpose on the spot
 * through an interface call).  This is a computational trick. We can
 * now figure out the GPU computation (column major) 
*/

void cpu_zgemm_gate_t (
     Index M, Index N, Index K,  
     ZC* AR, // vector  LB elements            
     ZC* BR, // square single matrix MxK stored in column major
     
     ZC* CR, Index LB,    // vector  LC = LB elements 
     Index batch) {

  /* notice we still do C= A*B but we stream C and A */
  ZC *B = BR;  /* this is the gate ... only one */
  ZC *A ;
  ZC *C;

  // We split I into batch parts
  unsigned int chunk = LB/batch; 
  unsigned int LD    = chunk/M; 
  
  if (N==1 or batch==1) LD=1;

  if (debug1) std::cout <<  "Chunk "<<  chunk << "LD " << LD << "\n";
  
  for (int p = 0; p < batch; ++p) {
    A = AR+ p*chunk; // chunk = MxLD
    C = CR+ p*chunk;

    
    //  C = B*U but U = KxN stored column major and
    //  logically/practically transpose
    
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
	ZC sum = ZERO;
	for (int k = 0; k < K; ++k){

	  /* the change is subtle but this is a classic GEMM and
	     column major ! we can call BLAS and BLAS
	   */ 

	  sum = sum +A[m +k*LD]*B[k+n*N];
	  if (debug1) std::cout <<  A[m +k*M]<< " * " << B[k*N+n]<<" = " << sum << "\n";
	}
	C[m+n*N] = sum;
	if (debug1) std::cout <<  " indx " <<  m+n*N << "<- " <<  sum << "\n";
      }
    }
    
  }
}
/*
 * This is as above A * B but we transpose and we actually compute the
 * B^t * A^t : this is cleaner when we use column major and A = A^t
 * (but we may just store the transpose or transpose on the spot
 * through an interface call).  This is a computational trick. We can 
 * now figure out the GPU computation (column major) 
*/

void gate::cpu_zgemm_matrix_gate_t ( 
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch) {

  
  Index LB = CR.M;
  Index chunk = LB/batch; // each small GEMM will compute chunks 
  Index LD    = chunk/BR.M; 
  
  for (Index p = 0; p < batch; ++p) {
    
    Matrix A{LD,BR.M,LD,BR.M, AR.matrix+ p*chunk}; 
    Matrix C{LD,BR.M,LD,BR.M, CR.matrix+ p*chunk}; 


    C.gemm(C,ZERO,A,BR,ONE);


  }
}


void gate::gpu_zgemm_matrix_gate_t (     rocblas_handle handle,
 
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch) {

  
  Index LB = CR.M;
  Index chunk = LB/batch; // each small GEMM will compute chunks 
  Index LD    = chunk/BR.M;
  AR.writetodevice();
  BR.writetodevice();
  CR.writetodevice();
  
  for (Index p = 0; p < batch; ++p) {
    
    Matrix A{LD,BR.M,LD,BR.M, AR.matrix+ p*chunk,AR.d_matrix+ p*chunk}; 
    Matrix C{LD,BR.M,LD,BR.M, CR.matrix+ p*chunk,CR.d_matrix+ p*chunk}; 


    C.gemm_gpu(C,ZERO,A,BR,ONE,handle);


  }
  CR.readfromdevice();
  
}

void gate::pre_gpu_gemm_t(
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch) {

  Index LB = CR.M;
  Index chunk = LB/batch; // each small GEMM will compute chunks 
  Index LD    = chunk/BR.M; 

  // We allocate pointers in the host and in the device 
  alloc(true,true);

  for (Index p = 0; p < batch; ++p) {
    h_A_ptrs[p] = AR.d_matrix+ + p * chunk;
    h_B_ptrs[p] = BR.d_matrix;
    h_C_ptrs[p] = CR.d_matrix+ p * chunk;
  }

  CHECK_HIP_ERROR(hipMemcpy(d_A_ptrs, h_A_ptrs, batch * sizeof(ZC*), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_B_ptrs, h_B_ptrs, batch * sizeof(ZC*), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_C_ptrs, h_C_ptrs, batch * sizeof(ZC*), hipMemcpyHostToDevice));




}

void gate::gpu_zgemm_matrix_gate_t_2 ( 
     rocblas_handle handle,
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch ) {



  Index LB = CR.M;
  Index chunk = LB/batch; // each small GEMM will compute chunks 
  Index LD    = chunk/BR.M; 

  Index M = LD;
  Index N = BR.N;
  Index K = BR.M; 
  CHECK_ROCBLAS_STATUS(
      rocblas_zgemm_batched(
	      handle, 
	      rocblas_operation_none, rocblas_operation_transpose, // Transpose options (None means no transpose)
	      M, N, K,
	      &ONE,
	      d_A_ptrs, M, // Pointer to array of const A pointers
	      d_B_ptrs, K, // Pointer to array of const B pointers
	      &ZERO,
	      d_C_ptrs, M, // Pointer to array of C pointers
	      batch
			    )
		);

  
}

 




void gate::init(Matrix I, Matrix O, int comp) {
  
  // already allocated nothing to
  U.alloc(true, true);
  U.writetodevice();
  U.transpose = true;
  
  if (gate::comp != comp) gate::comp=comp;
  
  // remember we are doing O = I* U (where U is U^t)
  
  int B = I.m; // this is the state 2^n 
  batch_count = B - ((1<<bit_number)*((bit_number==0)?0:1)
		     +U.m);
  
  batch_count = std::max(1, batch_count);
  printf("init\n");
  if (debug1)  U.print(true);
  
  
  m = U.m;
  n = 1<<bit_number;;
  k = U.n; 
  printf(" m %d n %d k %d \n", m, n, k);
  printf(" B %d batch_count %d bit_number %d \n", B, batch_count, bit_number);

  // if I and O are in place this can be done once 
  pre_gpu_gemm_t(I, U , O, batch_count);
  
}





void gate::step(rocblas_handle handle,
		Matrix &I, Matrix &O,
		int count ) {

  
  printf("######   Step %d \n", count );
  if (debug1) U.print();
  if (debug1) printf(" m %d n %d k %d bc %d \n", m, n, k,batch_count);
  
  const rocblas_stride strideA = m*k;
  const rocblas_stride strideB = k*n;
  const rocblas_stride strideC = m*n;
  
  if (debug1)  printf(" stridaA %ld strideB %ld strideC %ld \n",
		      strideA,strideB, strideC);
  
  U.print(true);
  if (debug1) O.print(true);

  switch (comp) {
  case 0:
    cpu_zgemm_matrix_gate_t(I,U,O,batch_count); 
    printf("CPU\n");
    break;
  case 1:

    gpu_zgemm_matrix_gate_t(handle,I,U,O,batch_count); 
    printf("GPU 1 \n");
    break;

  case 2:

    //   pre_gpu_gemm_t(I, U , O, batch_count);
    gpu_zgemm_matrix_gate_t_2(handle,I,U,O,batch_count); 
    
    printf("GPU 2 \n");
    break;
  default :
    cpu_zgemm_matrix_gate_t(I,U,O,batch_count); 
    printf("CPU\n");
  
  }

  if (debug1) printf("######   Step %d \n \n",count);
  
}




  // we move all the matrices into the 
void schedule::init(int comp){
    printf("Circuit init \n");
    for (std::vector<Gate> &level  : schedule)
      for (Gate &h : level ) { 
	h.init(I,O,comp);
      }
    printf("Circuit init \n\n");
  }

  
void schedule::forward(rocblas_handle handle) {
  int count = 0;
  printf("Circuit forward \n");
  I.print(true);

  int lvl =0 ;
  for (std::vector<Gate> &level  : schedule) { 
    
    
    for (Gate &h : level ) { 
      
      // We need to ping pong I and O 
      h.step(handle,
	     (lvl%2==0)?I:O,
	     (lvl%2==0)?O:I,
	     count++);

      if (true || debug1) { 
	if (lvl%2==0) {
	  if (h.comp>0) O.readfromdevice();
	  O.print(true);
	}
	else {
	  if (h.comp>0) I.readfromdevice();
	  I.print(true);
	}
      }
      lvl ++;
    }
  }
  printf("Circuit forward \n\n");
}

void schedule::forward_inplace(rocblas_handle handle) {
  int count = 0;
  printf("Circuit forward inplace \n");
  
  for (std::vector<Gate> &level  : schedule) { 
    
    
    for (Gate &h : level ) { 
      

      h.step(handle,I,I,count++);
      if (h.comp>0) {
	I.readfromdevice();
      }
      I.print(true);
      
    }
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
  ZERO, ONE,  ZERO, ZERO,
  ZERO, ZERO, ZERO, ONE,
  ZERO, ZERO, ONE,  ZERO}; 
Matrix CNM_{4,4,4,4, cnot_matrix};
Gate CNot{"CNot", CNM_};

	     


