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
static int debug2= 0;



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


    C.gemm_openblas(C,ZERO,A,BR,ONE);


  }
  ops = batch*LD*BR.m*BR.n*2*4;
}


void gate::gpu_zgemm_matrix_gate_t (     rocblas_handle handle,
 
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch) {

  
  Index LB = CR.M;
  Index chunk = LB/batch; // each small GEMM will compute chunks 
  Index LD    = chunk/BR.M;
  //AR.writetodevice();
  //BR.writetodevice();
  //CR.writetodevice();
  
  for (Index p = 0; p < batch; ++p) {
    
    Matrix A{LD,BR.M,LD,BR.M, AR.matrix+ p*chunk,AR.d_matrix+ p*chunk}; 
    Matrix C{LD,BR.M,LD,BR.M, CR.matrix+ p*chunk,CR.d_matrix+ p*chunk}; 


    C.gemm_gpu(C, &ZERO,// d_beta,
	       A,BR,
	       &ONE, //d_alpha,
	       handle);


  }
  ops = batch*LD*BR.m*BR.n*2*4;
  //  CR.readfromdevice();
  
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

  ops = batch*M*N*K*2*4;
  
  CHECK_ROCBLAS_STATUS(
     GEMM_BATCHED(
	      handle, 
	      rocblas_operation_none, rocblas_operation_transpose, // Transpose options (None means no transpose)
	      M, N, K,
	      &ONE, //d_alpha,
	      d_A_ptrs, M, // Pointer to array of const A pointers
	      d_B_ptrs, K, // Pointer to array of const B pointers
	      &ZERO, //d_beta,
	      d_C_ptrs, M, // Pointer to array of C pointers
	      batch
			    )
		);

  //CHECK_HIP_ERROR(hipDeviceSynchronize());
}

 




void gate::init(Matrix I, Matrix O, int comp) {
  
  // already allocated nothing to
  U.alloc(true, true);
  U.writetodevice();
  U.transpose = true;
  
  if (gate::comp != comp) gate::comp=comp;
  
  // remember we are doing O = I* U (where U is U^t)
  
  int B = I.m; // this is the state 2^n
  int K = 1<<bit_number;
  batch_count = B/ (K*U.m);
  

  batch_count = std::max(1, batch_count);
  if (batch_count==42) {
    printf("WHATTA \n");
    printf("%f",1.0/0.0); } 
  if (debug1) printf("init\n");
  if (debug1)  U.print(true);
  
  
  m = U.m;
  n = 1<<bit_number;;
  k = U.n; 
  if (debug1) printf(" m %d n %d k %d \n", m, n, k);
  if (debug1) printf(" B %d batch_count %d bit_number %d \n", B, batch_count, bit_number);

  // if I and O are in place this can be done once 
  pre_gpu_gemm_t(I, U , O, batch_count);
  
}





void gate::step(rocblas_handle handle,
		Matrix &I, Matrix &O,
		int count ) {

  auto start_ = std::chrono::high_resolution_clock::now();
  
  if (debug1) printf("######   Step %d \n", count );
  if (debug1) U.print();
  if (debug1) printf(" m %d n %d k %d bc %d \n", m, n, k,batch_count);
  
  
  if (debug1) U.print(true);
  if (debug1) O.print(true);
  calls++;
  switch (comp) {
  case 0:
    cpu_zgemm_matrix_gate_t(I,U,O,batch_count); 
    if (debug1) printf("CPU\n");
    break;
  case 1:

    gpu_zgemm_matrix_gate_t(handle,I,U,O,batch_count); 
    if (debug1) printf("GPU 1 \n");
    break;
 
  case 2:
    //print(true);
    //   pre_gpu_gemm_t(I, U , O, batch_count);
    gpu_zgemm_matrix_gate_t_2(handle,I,U,O,batch_count); 
    
    if (debug1)  printf("GPU 2 \n");
    break;
  default :
    print(true);
    I.print(true);
      
    /* we can use this to validate all of the above ? */
    Matrix C{I.m, I.n, I.M, I.N};
    C.alloc(true, false);
    Matrix G1{I.m, I.n, I.M, I.N};
    G1.alloc(true, true);
    Matrix G2{I.m, I.n, I.M, I.N};
    G2.alloc(true, true);
      
    cpu_zgemm_matrix_gate_t(I,U,C,batch_count); 
    C.print(true);
    gpu_zgemm_matrix_gate_t(handle,I,U,G1,batch_count);
    G1.readfromdevice();
    G1.print(true);
    
    pre_gpu_gemm_t(I, U , G2, batch_count);
    gpu_zgemm_matrix_gate_t_2(handle,I,U,G2,batch_count);
    G2.readfromdevice();
    G2.print(true);

    C.free();
    G1.free();
    G2.free();
    
    if (debug1) printf("DEBUG \n");
  
  }

  if (debug1) printf("######   Step %d \n \n",count);
  auto end_ = std::chrono::high_resolution_clock::now();
  
  // 3. Calculate duration (e.g., in microseconds)
  auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>((end_ - start_));
  if (execution_time!=0) execution_time = std::min(execution_time,duration_.count()/1000000000.0);
  else execution_time=duration_.count()/1000000000.0;
  tflops = ops/execution_time/1000000000000;
}




  // we move all the matrices into the 
void schedule::init(int comp){
    if (debug1) printf("Circuit init \n");
    for (std::vector<Gate> &level  : schedule)
      for (Gate &h : level ) { 
	h.init(I,O,comp);
      }
    if (debug1) printf("Circuit init \n\n");
  }

  
void schedule::forward(rocblas_handle handle) {
  int count = 0;
  if (debug1) printf("Circuit forward \n");
  I.print(true);

  int lvl =0 ;
  for (std::vector<Gate> &level  : schedule) { 
    
    
    for (Gate &h : level ) { 
      
      // We need to ping pong I and O 
      h.step(handle,
	     (lvl%2==0)?I:O,
	     (lvl%2==0)?O:I,
	     count++);

      if (debug1) { 
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
  if (debug1) printf("Circuit forward \n\n");
}

void schedule::forward_inplace(rocblas_handle handle) {
  int count = 0;
  if (debug2)  printf("Circuit forward inplace \n");
  if (debug2)  I.print(true);
  
  ops=0;
  
  for (std::vector<Gate> &level  : schedule) { 
    long long int lops =0;
    auto start = std::chrono::high_resolution_clock::now();
    for (Gate &h : level ) { 

      h.step(handle,I,I,count++);
      lops += h.ops;      
      if (debug2) { 
	if (h.comp>0) {
	  I.readfromdevice();
	}
	I.print(true);
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    // 3. Calculate duration (e.g., in microseconds)
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start));
    double level_time =  duration.count()/1000000000.0;
    std::cout << "Time: " << level_time << " Ops" << lops << 
      " TFlops: " << lops/level_time/1000000000000 << std::endl;

    ops =lops;
  }
  

  if (debug2)  printf("Circuit forward \n\n");
}



void schedule::print(bool  t)  {
    int l =0;
    printf("BEGIN Circuit %zu \n", schedule.size());
    I.print(true);
    for (std::vector<Gate> &level  : schedule) {
      printf("Level %d < %zu \n", l++, level.size());
      int c =0;
      for (Gate &h : level )  { 
	if (c<5 ||  c>level.size()-5)
	  h.print(t);
	c++;
      }
      
    }
    printf("END Circuit \n");
  }










// Construction of gates
static NORM_TYPE SQ2 = 1.0/std::sqrt(2);

const NORM_TYPE PI_8_COS = std::cos(M_PI / 4.0); // For T-gate
const NORM_TYPE PI_8_SIN = std::sin(M_PI / 4.0);


/** 1/sqrt(2) | 1  1 |
 *            | 1 -1 | 
 * Stored in column major 
 */

alignas(64) static ZC hadamard_matrix[] =  {  SQ2*ONE, SQ2*ONE, SQ2*ONE,-SQ2*ONE }; 
Matrix HM_{2,2,2,2, hadamard_matrix};
Gate Hadamard{"hadamard", HM_};

/**  | 1  0 |
 *   | 0  1 | 
 * Stored in column major but complex numbers  
 */
alignas(64) static ZC identity_matrix[] =  {  ONE, ZERO, ZERO, ONE }; 
Matrix IM_{2,2,2,2, identity_matrix};
Gate Identity{"Identity",IM_};

/**  | 0 1 |
 *   | 1 0 | 
 * Stored in column major but complex numbers  x() or not
 */
alignas(64) static ZC pauli_x_matrix[] =  {  ZERO, ONE, ONE, ZERO };
Matrix PXM_{2,2,2,2, pauli_x_matrix};
Gate Pauli_X{"PauliX", PXM_};

/**  | 0  -i |
 *   | i  0 | 
 * Stored in column major but complex numbers  
 */

alignas(64) static ZC pauli_y_matrix[] =  { ZERO, 1.0i*ONE, -1.0i* ONE, ZERO  }; 
Matrix PYM_{2,2,2,2, pauli_y_matrix};
Gate Pauli_Y{"PauliY",PYM_};

/**  | 1  0 |
 *   | 0  -1 | 
 * Stored in column major but complex numbers  
 */
alignas(64) static const ZC pauli_z_matrix[] =  { ONE, ZERO, ZERO, -ONE }; 
Matrix PZM_{2,2,2,2, pauli_x_matrix};
const Gate Pauli_Z{"PauliZ", PZM_};



/**  | 1 0 0 0 |
 *   | 0 1 0 0 |
 *   | 0 0 0 1 |
 *   | 0 0 1 0 |
 
 * Stored in column major but complex numbers  
 */
alignas(64) static  ZC cnot_matrix[] =  {
  ONE,  ZERO, ZERO, ZERO,
  ZERO, ONE,  ZERO, ZERO,
  ZERO, ZERO, ZERO, ONE,
  ZERO, ZERO, ONE,  ZERO}; 
Matrix CNM_{4,4,4,4, cnot_matrix};
Gate CNot{"CNot", CNM_};

	     

alignas(64) static ZC toffoli_matrix[] {
    ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, // Column ONE
    ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, // Column 2
    ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, // Column 3
    ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, // Column 4
    ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, // Column 5
    ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, // Column 6
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, // Column 7
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO  // Column 8
};
Matrix CCNM_{8,8,8,8, toffoli_matrix};
Gate CCNot{"CCNot", CNM_};

alignas(64) static ZC fredkin_matrix[]{
    ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, // Column ZERO: |ZEROZEROZERO>
    ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, // Column ONE: |ZEROZEROONE>
    ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, // Column 2: |ZEROONEZERO>
    ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, // Column 3: |ZEROONEONE>
    ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, // Column 4: |ONEZEROZERO>
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, // Column 5: |ONEZEROONE> -> Swaps to |ONEONEZERO>
    ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, // Column 6: |ONEONEZERO> -> Swaps to |ONEZEROONE>
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE  // Column 7: |ONEONEONE>
};
Matrix FM_{8,8,8,8,  fredkin_matrix};
Gate Fredkin{"fredkin",FM_} ;

// T-Gate (pi/8 gate)
alignas(64) static ZC t_matrix[] = {
  ONE, ZERO,                // Column 0
  ZERO, ZC{PI_8_COS, PI_8_SIN}  // Column 1
};
Matrix TM_{8,8,8,8,  t_matrix};
Gate T{"T",TM_} ;

static const NORM_TYPE phi = (1.0 + std::sqrt(5.0)) / 2.0; // 1.618...
static const NORM_TYPE inv_phi = 1.0 / phi;               // 0.618...

inline ZC exp_i(NORM_TYPE theta) {
    return ZC{ std::cos(theta), std::sin(theta) };
}




// 2x2 Symmetric Matrix
// A[0][1] == A[1][0]
alignas(64) static ZC MTWO[]{
  exp_i(phi),    exp_i(inv_phi),
  exp_i(inv_phi),  exp_i(-phi)
};
Matrix TMTWO_{2,2,2,2,  MTWO};
Gate TTwo{"T2x2",TMTWO_} ;

// 4x4 Symmetric Matrix
// Pattern uses powers of phi for "randomness" while maintaining A[i][j] = A[j][i]
alignas(64) static ZC  MFOUR[]{
  exp_i(phi ),        exp_i(inv_phi * 2),  exp_i(phi * 3),      exp_i(inv_phi * 4),
  exp_i(inv_phi * 2), exp_i(-phi * 2),     exp_i(inv_phi * 5),  exp_i(phi * 6),
  exp_i(phi * 3),     exp_i(inv_phi * 5),  exp_i(phi * 3),      exp_i(inv_phi * 7),
  exp_i(inv_phi * 4), exp_i(phi * 6),      exp_i(inv_phi * 7),  exp_i(-phi * 4)
};
Matrix TMFOUR_{4,4,4,4,  MFOUR};
Gate TFour{"T4x4",TMFOUR_} ;



