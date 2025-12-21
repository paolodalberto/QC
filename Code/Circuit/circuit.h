#pragma once 

/***
 * For batched computation we need to prepare the pointers
 */

extern 
void pre_gpu_gemm(
     int M, int N, int K, 
     ZC** A, rocblas_stride ldA, ZC *d_A,
     ZC** B, rocblas_stride ldB, ZC *d_B,
     ZC** C, rocblas_stride ldC, ZC *d_C,
     int batchCount
		  );
/***
 * For batched computation we need to prepare the pointers
 */

extern 
void pre_gpu_gemm_B(
     int M, int N, int K, 
     ZC** A, rocblas_stride ldA, ZC *d_A,
     ZC** B, rocblas_stride ldB, ZC *d_B,
     ZC** C, rocblas_stride ldC, ZC *d_C,
     int batchCount
		  );
/***
 * After the pre you can run the execute 
 */
extern 
void gpu_zgemm_batched(
     rocblas_handle handle,
     int M, int N, int K, ZC alpha, 
     ZC** A, rocblas_stride ldA,
     ZC** B, rocblas_stride ldB,
     ZC beta,
     ZC** C, rocblas_stride ldC,
     int batchCount
		       );



/***************
 **  A circuit is a sequence of layers. This defines the depth of the
 **  circuit. The layers is a sequence/column gates.
 **
 **  Each gate has the property that the order does not matter but
 **  they are NOT parallel.  Again take a look at the notes and the
 **  python implementation. There is an opportunity to fuse Gates to
 **  reduce the number of passes throught the state but this will
 **  affect the "rows" of the computation and thus the interference in
 **  caches.
 **
 ********/



struct gate {

  /* Every one deserves a unique name so if you are using the same
   *  gate identified by a name we can do this quite easily and we do
   *  not need to have multiple copies .. there will be only one gate
   *  but it will be applied to different bits thus the computation
   *  will be different
  */

  std::string name;  
  Matrix &U; // Gate matrix gxg : shared and already transposed   
  

  Index bit_number=0; // the first bit where we apply the gate k
                   

  Matrix I; // Input state : 2^n x 1: shared 
  Matrix O; // output state         : shared and this can be the input state  

  Index m=0; // index of there we apply the gate
  Index n=0; // kernel size and this is G kernel (mxk)
  Index k=0;

  Index batch_count =1;
  ZC &alpha = ALPHA;
  ZC &beta  = BETA; 

  // host pointers the strided pointers for the computation in the host 
  ZC **h_A_ptrs =0 ;
  ZC **h_B_ptrs =0 ;
  ZC **h_C_ptrs =0 ;

  // device pointers  as above 
  ZC **d_A_ptrs = 0 ;
  ZC **d_B_ptrs =0 ;
  ZC **d_C_ptrs =0;

  int comp = 0 ; // 0 CPU 1 GPU
  
  Index index( Index bit) {return 1<<bit;}
  void set_index(Index bit) {
    bit_number = bit;
    m  =  1<<bit;
  }

  
  // We allocate the pointers only  
  void alloc(bool host, bool device) {
    if (host) { 
      h_A_ptrs = (ZC**)std::malloc(batch_count * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_A_ptrs did not make it");
      h_B_ptrs = (ZC**)std::malloc(batch_count * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_B_ptrs did not make it");
      h_C_ptrs = (ZC**)std::malloc(batch_count * sizeof(ZC*));
      assert(h_A_ptrs!=0 && " h_C_ptrs did not make it");
    }
    if (device ) {
      CHECK_HIP_ERROR(hipMalloc((void**)&d_A_ptrs, batch_count * sizeof(ZC*)));
      CHECK_HIP_ERROR(hipMalloc((void**)&d_B_ptrs, batch_count * sizeof(ZC*)));
      CHECK_HIP_ERROR(hipMalloc((void**)&d_C_ptrs, batch_count * sizeof(ZC*)));
    }
  }

  // we free the pointers and the gate
  void free() {
    
    if (h_A_ptrs) { std::free(h_A_ptrs); h_A_ptrs=0;} 
    if (h_B_ptrs) { std::free(h_B_ptrs); h_B_ptrs=0;} 
    if (h_C_ptrs) { std::free(h_C_ptrs); h_C_ptrs=0;} 

    CHECK_HIP_ERROR(hipFree(d_A_ptrs));
    CHECK_HIP_ERROR(hipFree(d_B_ptrs));
    CHECK_HIP_ERROR(hipFree(d_C_ptrs));

    // gate
    U.free();
  }
  
  /* This is per object: A Gate per bit location.
   * We share the Input and the output
   * we share the matrix of the gate
   */
  void init(Matrix II, Matrix OO, int comp=0);
  void step(rocblas_handle handle,
	    Matrix &I, Matrix &O, int count =0);

  void cpu_zgemm_matrix_gate_t ( 
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch);

  // multiple call to gpu GEMM
  void gpu_zgemm_matrix_gate_t (
     rocblas_handle handle,
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch);

  // Multiple addresses and single call 
  void gpu_zgemm_matrix_gate_t_2 (
     rocblas_handle handle,
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch);


 void pre_gpu_gemm_t(
     Matrix &AR,   // vector  LB elements
     Matrix &BR,   // square single matrix MxM stored in column major
     Matrix &CR,   // vector  LC = LB elements 
     Index batch);  

  
  void print(bool t=false) {
    std::cout << ">>>>>> Gate " <<  name << "\n"; 
    printf("Bit %d \n", m);
    printf("Batch %d \n",batch_count );
    U.print(t);
    std::cout << "<<<<<< Gate " <<  name << "\n"; 

  }

  

};

// When you see a Gate is a struct     
typedef struct gate Gate;



struct schedule {
  Matrix &I;  // Input state 
  Matrix &O;  // Input state 

  /*
   * the depth of the circuit is the lenght of the outside vector the
   * inner vector is the vertical collection of gates
   */
  std::vector<std::vector<Gate>> &schedule; 
  
  // we move all the matrices into the 
  void init(int comp=0);
  void forward(rocblas_handle handle);
  void forward_inplace(rocblas_handle handle);
  void print(bool  t=false);
};

typedef struct schedule Circuit;
