
#pragma once

#include "type.h"
// I like to wrap the matrices into a container
template <typename Entry>
struct matrix {
  
  // struct as a class members are all public
  Index m;                       // rows
  Index n;                       // cols
  Index M;                       // Maximum rows LD
  Index N;                       // Maximum cols
  Entry *matrix = 0;             // host  
  Entry *d_matrix =0;            // device
  bool gate = true;
  bool transpose= false;
  bool conjugate_transpose= false;
  int gpu = -1;
  rocblas_handle handle_gpu=0;
  
  void set_device() {
    if (gpu!=-1)  CHECK_HIP_ERROR(hipSetDevice(gpu));
  }
  
  size_t initialize_host_matrix(const Entry* initial_data) {
    // Calculate total size needed
    size_t total_elements = m * n;
    size_t total_bytes = total_elements * sizeof(Entry);
    
    // 1. Allocate the memory using malloc
    alloc(true,false);
    
    if (matrix == nullptr) {
      std::cerr << "Memory allocation failed!" << std::endl;
      return 0;
    }
    
    memcpy(matrix, initial_data, total_bytes);
    return total_bytes;
  }
  
  void free() {
    if (matrix and !gate)   { std::free(matrix);        matrix =0; }
    if (d_matrix) {
      set_device();
      CHECK_HIP_ERROR(hipFree(d_matrix));
      d_matrix=0;}
  }
  void alloc(bool host , bool device  ) {
    if (size()>0) {
      //printf(" Allocated %d * %d = %d elements \n", M, N,M*N);
      if (host and matrix==0)   {
	matrix = (Entry*) std::calloc(M*N,sizeof(Entry));
	assert(matrix!=0 && " Failed to allocate Doh\n");
      }
      if (device and d_matrix==0) {
	set_device(); 
	CHECK_HIP_ERROR(hipMalloc(&d_matrix, M*N* sizeof(Entry)));
      }		      
    }
  }
  void readfromdevice() {
    if ( matrix!=0 and d_matrix!=0) { 
      set_device(); 
      CHECK_HIP_ERROR(hipMemcpy(matrix , d_matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));
      
    }
  }
  void writetodevice() {
    if ( matrix!=0 and d_matrix!=0)  {
      set_device(); 
      CHECK_HIP_ERROR(hipMemcpy(d_matrix , matrix, size() * sizeof(Entry), hipMemcpyHostToDevice));
    }
  }
  
  /**********
   *  These four operations determines the * + algebra in hte CPU and
   *  GPU so if you like you can replicate the computation or perform
   *  CPU standalone 
   *  
   *  Remeber CPU (*,+) -> GPU (*,+) -> CPU (*,+) 
   */ 
  
  void gemm(struct matrix &C, Entry beta, struct matrix &A,
	    struct matrix &B, Entry alpha, const int debug1=0){
    struct matrix T{C.m, C.n, C.M,C.N};
    T.alloc(true, false);
    for (Index m = 0; m < C.m; ++m) 
      for (Index n = 0; n < C.n; ++n) {
      ZC sum = ZERO;
      
      for (Index k = 0; k < A.n; ++k) { 
	sum = sum +A.matrix[A.ind(m,k)]*B.matrix[B.ind(k,n)];
	if (debug1) std::cout << A.matrix[A.ind(m,k)] << " * " << B.matrix[B.ind(k,n)]<<" = " << sum << "\n";
      }
      T.matrix[T.ind(m,n)] = alpha*sum;
      
      if (debug1) std::cout <<  " indx " << C.ind(m,n)  << "<- " <<  sum << "\n";
      }
    
    for (Index m = 0; m < C.m; ++m) 
      for (Index n = 0; n < C.n; ++n) 
	C.matrix[C.ind(m,n)] = T.matrix[T.ind(m,n)]  +  C.matrix[C.ind(m,n)]*beta;
    T.free();
    
  }
  void geam(struct matrix &C, Entry beta, struct matrix &A,
	    struct matrix &B, Entry alpha, const int debug1=0) {
    for (Index m = 0; m < C.m; ++m) 
      for (Index n = 0; n < C.n; ++n) {
	C.matrix[C.ind(m,n)] = alpha*(A.matrix[A.ind(m,n)] + B.matrix[B.ind(m,n)]) +  C.matrix[C.ind(m,n)]*beta;
      }
  }
  void gemm_openblas(struct matrix &C, Entry beta, struct matrix &A,
	    struct matrix &B, Entry alpha, const int debug1=0){
    struct matrix T{C.m, C.n, C.M,C.N};
    T.alloc(true, false);
    GEMMC(CblasColMajor, // Use Column Major
	  CblasNoTrans,  // Op(A) = A
	  CblasNoTrans,  // Op(B) = B
	  A.m, B.n, B.n,  // this is the problem size                                                 
	  &alpha, 
	  A.matrix, A.m, 
	  B.matrix, B.m, 
	  &beta, 
	  T.matrix, T.m);
    
    for (Index m = 0; m < C.m; ++m) 
      for (Index n = 0; n < C.n; ++n) 
	C.matrix[C.ind(m,n)] = T.matrix[T.ind(m,n)]  +  C.matrix[C.ind(m,n)]*beta;
    T.free();
    
  }
  
  void gemm_gpu(struct matrix &C, Entry* beta, struct matrix &A,
		struct matrix &B, Entry* alpha,
		rocblas_handle handle=0)  {
    // C = alpha A * B + beta C
    /*
      The transposistion is a computation trick to exploit the column
      major representation of the gate matrices, for complex matrices
      and quantum computing this is more often a conjugate transpose .
      for a gate O = G I we do actually compute O^t I^ G^t but G is
      already column major and thus is physically transpose ? and I is
      a vector ...
    */

    CHECK_ROCBLAS_STATUS(
	GEMM(handle_gpu, 
	     rocblas_operation_none,
	     rocblas_operation_none,
	     A.m, B.n, B.n,  // this is the problem size                                                 
	     alpha, 
	     A.d_matrix, A.m, 
	     B.d_matrix, B.m, 
	     beta, 
	     C.d_matrix, C.m)); 
  
  }
  void geam_gpu(struct matrix &C, Entry beta, struct matrix &A,
		struct matrix &B, Entry alpha, rocblas_handle handle=0)  {
  
    // C = alpha (A + B) + beta C
    CHECK_ROCBLAS_STATUS(
	GEAM(handle, 
	     rocblas_operation_none,
	     rocblas_operation_none,
	     A.m, B.n, B.n,  // this is the problem size                                                 
	     &alpha, 
	     A.d_matrix, A.m, 
	     B.d_matrix, B.m, 
	     &beta, 
	     C.d_matrix, C.m)); 
  }



  void init() {
      for (Index i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(i % 10);
  };
  void zero() {
    for (Index i = 0; i < m * n; ++i) matrix[i] = static_cast<Entry>(0);
  };
  void bra_zero() {
    zero();
    matrix[0] = ONE;
  };
  Index ind(Index i, Index j, bool t=false)    {
    if (t || transpose )
      return i*n +j;
    else
      return i +m*j;
    
  }
  
  size_t size() { return m*n; } 
  void print(bool t=false) {
    Index MM = (m>10)?10: m;
    Index NN = (n>10)?10: n;
    std::cout << "Column Major M,N = "<< m << "," << n << "\n";
    if (t)
      for (Index i = 0; i < MM; ++i) {
	for (Index j = 0; j < NN; ++j) 
	  std::cout << matrix[ind(i,j)] << " " ;
	printf("\n");
      }
  };
};


typedef struct matrix<ZC> Matrix;
typedef struct matrix<NORM_TYPE> EigenValueMatrix;
  

  
struct buffer { // this is a device buffer
  int gpu;
  // State
  ZC     *state=0;   // matrix.d_matrix pointer 
  size_t s_bytes=0;
  
  // relative pointer where to read in the state
  ZC    *origin=0;

  // temporary buffer
  ZC    *temp=0;
  size_t t_bytes=0;

  void free() {
    if (0 and state) { CHECK_HIP_ERROR(hipFree(state)); state=0;} 
    if (temp)  { CHECK_HIP_ERROR(hipFree(temp)); temp=0;} 
  };

  void print() {
    printf(" Buffer at GPU %d State %p Size %ld Origin %p AND Temp %p  Size %ld \n",
	   gpu, (void*)state,s_bytes, (void*)origin, (void*)temp,  t_bytes);
  }
  
};


struct connection  {
  struct buffer &A;
  struct buffer &B;
  hipStream_t   s=0;
  int can_access=0;

  void free() {
    A.free();
    B.free();
    if (s) { CHECK_HIP_ERROR(hipStreamDestroy(s)); s = 0; }
  };
  void print() {
    printf("Source : ");
    A.print();
    printf("Dest   : ");
    B.print();
    printf("Stream Handle Value: %p Peer %d\n", (void*)s, can_access);
  }

};



// This kernel represents the "Rewrite" phase after the transfer.
// It places the incoming data into the correct bit-permuted location.


/*
 A memory block is specified by a GPU and an integer identifying the
 relative block ..this is a small number describing how many bits we
 are swapping and thus a simplified computation of the relative
 address.

 This can be the source we want to transfer this block from this GPU
 somewhere else or it can be a destination usually an allocated space
 and we copy into a specific location and then moved .. the last part
 is important 

*/

struct memory_block {
  int   gpu;
  int   block;
  ZC     *state=0;   // matrix.d_matrix pointer 
  ZC     *temp =0;   // matrix.d_matrix pointer 
  size_t s_bytes=0;
  bool   temporary = false; // destination will use a temporary 

  void alloc(size_t size, ZC *reference=0) {
    if (reference!=0) { 
      s_bytes = size;
      state = reference + block*s_bytes/sizeof(ZC);
    }
    else {
      state = (ZC*) malloc(s_bytes);
    }
  };
  void free() {
    if (temporary and state) { std::free(state); state=0;}
  };

  void print() {
    printf(" Buffer at GPU %d block %d state %p Size %ld  \n",
	   gpu, block, (void*)state,s_bytes);
  }
  
  
};

typedef struct memory_block Block;

/* 
   Destination -> Source we will create a hip stream for each
   connection (gpu-gpu) this is a copy to the same hpstream connection


   Communication &c : P
   
   CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&c.can_access,
					   c.source.gpu,
					   c.destination.gpu));
   CHECK_HIP_ERROR(hipSetDevice(c.source.gpu));
   if (ONE.can_access) {
      CHECK_HIP_ERROR(hipDeviceEnablePeerAccess(c.destination.gpu, 0));
   }
   CHECK_HIP_ERROR(hipStreamCreate(&c.s)); 

   there should be a one stream and one can_access per gpu-gpu pair I
   believe we can transfer different blocks


 */



#include "network.h"

struct communication {
  Block source;
  Block destination;

  ConnStream &peer; /* these are per gpu-gpu conection*/

  void print() {
    printf("Can %d S:\n",peer.can_access);
    source.print();
    printf("D:\n");
    destination.print();
  };
};

typedef struct communication Communication;


typedef  std::vector<Communication>  Permutation;





template <typename Entry>
struct distributed_state{
  int bits;
  std::vector<Matrix> &G;
  BiGraph graph;
  
  std::vector<int> gpus;
  Index size=0; 

  void set_peers();
  void init();
  void free();
  void set_temp(std::vector<int> gpus,
		std::vector<int> permutation,
		std::vector<struct buffer> &Bs,
		std::vector<struct connection> &Cs,
		size_t state_bytes,size_t temp_bytes  );

  void run_shuffle_test(std::vector<int> permutation);
  void run_shuffle_test(Permutation &P);
  
};



typedef struct distributed_state<ZC> STATE;
