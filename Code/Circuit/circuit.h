
/**********
 * Reference computation in basic c++
 * using regular pointers
 *
 */ 

extern
void cpu_zgemm_batched_b(
     int M, int N, int K, ZC alpha, 
     ZC* A, rocblas_stride ldA,
     ZC* B, rocblas_stride ldB,
     ZC beta,
     ZC* C, rocblas_stride ldC,
     int batchCount
			 );
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




/*****
 * this is column major and thus we will need to compute O = I * G^t
 * but I will transpose directly G  ...
 ***/

void cpu_zgemm_batched_M(
     int M, int N, int K, ZC alpha, 
     Matrix &A,
     Matrix &B,  // B is the small one the gate one 
     ZC beta,
     Matrix &C,
     int batchCount) {
  
  cpu_zgemm_batched_b(
	   A.m, B.m, A.n, alpha, 
	   A.matrix, A.m, // I 
	   B.matrix, B.m, // G^t
	   beta,
	   C.matrix, C.m, // O
	   batchCount
		      );
  
}





/***************
 **  A circuit is a sequence of layers. This define the depth of the
 **  circuit. The layers is a sequence gates.  

 **  Each gate has the property that the order does not matter but
 **  they are NOT parallel.  Again take a look at the notes and the
 **  python implementation. There is an opportunity to fuse Gates to
 **  reduce the number of passes throught the state but this will
 **  affect the "rows" of the computation and thus the interference in
 **  caches.
 **
 **
 ********/



struct gate {

  /* Every one deserves a unique name so if you are
   using the same gate identified by a name we
   can do this quite easily and we do not need to
   have multiple copies .. there will be only one
   gate but it will be applied to different bits
   thus the computation will be different
  */
  std::string name;  


  int    bit_number; // the first bit where we apply the gate k
                   

  Matrix &I; // Input state : 2^n x 1: shared 
  Matrix &U; // Gate matrix gxg      : shared and already transposed   
  Matrix &O; // output state         : shared and this can be the input state  

  int m=0; // index of there we apply the gate
  int n=0; // kernel size and this is G kernel (mxk)
  int k=0;

  int batch_count =1;
  ZC alpha = ALPHA;
  ZC beta  = BETA; 

  // host pointers the strided pointers for the computation in the host 
  ZC **h_A_ptrs =0 ;
  ZC **h_B_ptrs =0 ;
  ZC **h_C_ptrs =0 ;

  // device pointers  as above 
  ZC **d_A_ptrs = 0 ;
  ZC **d_B_ptrs =0 ;
  ZC **d_C_ptrs =0;


  
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
  void init() {
    // U is square and we allocate in the host and the device if it is
    // already allocated nothing to do
    U.alloc(true, true);

    // remember we are doing O = I* U (where U is U^t)
    
    int B = I.m; // this is the state 2^n 
    batch_count = B - ((1<<bit_number)+U.m);

    m = 1<<bit_number;
    n = U.n; 
    k = U.m;

    alloc(true,true);
    pre_gpu_gemm(m,n,k,
		 h_A_ptrs,m,U.matrix,
		 h_B_ptrs,ldB,I.matrix,
		 h_C_ptrs,ldC,O.matrix,
		 batch_count);
    
  }
  
  
  
  void step(rocblas_handle handle) {
    
    const rocblas_stride strideA = m*n;
    const rocblas_stride strideB = k*n;
    const rocblas_stride strideC = n * n;
    const size_t total_elements_A = strideA * batch_count;
    const size_t total_elements_B = strideB * batch_count;
    const size_t total_elements_C = strideC * batch_count;
    
    
    CHECK_ROCBLAS_ERROR(
	gpu_zgemm_batched(
	    handle,
	    m, n, k, alpha, 
	    d_A_ptrs, strideA,
	    d_B_ptrs, strideB,
	    beta,
	    d_C_ptrs, strideC,
	    batch_count
			  );
	
			)
      }
};

// When you see a Gate is a struct     
typedef struct gate Gate;



struct schedule {
  Matrix &I;  // Input state 
  Matrix &O;  // Output state

  /*
   * the depth of the circuit is the lenght of the outside vector the
   * inner vector is the vertical collection of gates
   */
  std::vector<std::vector<Gate>> schedule; 
  
  // we move all the matrices into the 
  void init(){
    for (std::vector<Gate> &level  : schedule)
      for (Gate h : level )
	h.init();
    
  }
  
  
  void forward(rocblas_handle handle) {
    for (std::vector<Circuit> &level  : schedule)
      for (Circuit h : level )
	h.step(handle);
  }
};

typedef struct schedule Circuit;
