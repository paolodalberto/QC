#pragma once 



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
  Matrix &U; // Gate matrix gxg      : shared and already transposed   
  

  int    bit_number=0; // the first bit where we apply the gate k
                   

  Matrix I; // Input state : 2^n x 1: shared 
  Matrix O; // output state         : shared and this can be the input state  

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

  int index( unsigned int bit) {return 1<<bit;}
  void set_index(int bit) {
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
  void init(Matrix II, Matrix OO) {
    I = II; 
    O = OO; 
    // U is square and we allocate in the host and the device if it is
    // already allocated nothing to
    U.alloc(true, true);

    // remember we are doing O = I* U (where U is U^t)
    
    int B = I.m; // this is the state 2^n 
    batch_count = B - ((1<<bit_number)*((bit_number==0)?0:1)
		       +U.m);

    printf("init\n");
    U.print(true);
 

    m = 1<<bit_number;
    n = U.n; 
    k = U.m;
    printf(" m %d n %d k %d \n", m, n, k);
    printf(" B %d batch_count %d bit_number %d \n", B, batch_count, bit_number);
    
    alloc(true,true);
    pre_gpu_gemm(m,n,k,
		 h_A_ptrs,m,  I.d_matrix,
		 h_B_ptrs,U.m,U.d_matrix,
		 h_C_ptrs,m,  O.d_matrix,
		 batch_count);
   
    CHECK_HIP_ERROR(hipMemcpy(d_A_ptrs, h_A_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B_ptrs, h_B_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C_ptrs, h_C_ptrs, batch_count * sizeof(ZC*), hipMemcpyHostToDevice));
    
  }
  
  void print(bool t=false) {
    std::cout << ">>>>>> Gate " <<  name << "\n"; 
    printf("Bit %d \n", m);
    printf("Batch %d \n",batch_count );
    U.print(t);
    std::cout << "<<<<<< Gate " <<  name << "\n"; 

  }

  
  void step(rocblas_handle handle) {

    Matrix Z{4,1,4,1};
    Z.alloc(true,false);
    printf("######   Step \n");
    U.print();
    printf(" m %d n %d k %d bc %d \n", m, n, k,batch_count);
    
    const rocblas_stride strideA = m*n;
    const rocblas_stride strideB = k*n;
    const rocblas_stride strideC = n * n;
    const size_t total_elements_A = strideA * batch_count;
    const size_t total_elements_B = strideB * batch_count;
    const size_t total_elements_C = strideC * batch_count;
    
    I.print(true);
    U.print(true);
    Z.print(true);
      
    cpu_zgemm_batched_b(
		      m, n, k, alpha, 
		      I.matrix, strideA,
		      U.matrix, strideB,
		      beta,
		      Z.matrix, strideC,
		      batch_count);

    printf("Z cpu \n");
    Z.print(true);

    
    gpu_zgemm_batched(
	handle,
	m, n, k, alpha, 
	d_A_ptrs, strideA,
	d_B_ptrs, strideB,
	beta,
	d_C_ptrs, strideC,
	batch_count
		      );
    
    printf("######   Step \n \n");

      
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
  std::vector<std::vector<Gate>> schedule; 
  
  // we move all the matrices into the 
  void init(){
    printf("Circuit init \n");
    for (std::vector<Gate> &level  : schedule)
      for (Gate &h : level ) { 
	h.init(I,O);
      }
    printf("Circuit init \n\n");
  }

  
  void forward(rocblas_handle handle) {
    printf("Circuit forward \n");
    for (std::vector<Gate> &level  : schedule)
      for (Gate h : level ) { 
	I.print(true);
	I.writetodevice();
	h.step(handle);
	O.readfromdevice();
	O.print(true);
      }
    printf("Circuit forward \n\n");
  }
  void print(bool  t=false)  {
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
};

typedef struct schedule Circuit;
