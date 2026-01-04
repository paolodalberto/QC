#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define CHECK_HIP(expr) { \
    hipError_t status = (expr); \
    if(status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorName(status) \
                  << " (" << status << ")" << std::endl \
                  << "Message: " << hipGetErrorString(status) << std::endl \
                  << "Location: " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("HIP Runtime Error"); \
    } \
}
//#define CHECK_HIP(expr) { hipError_t status = (expr); if(status != hipSuccess) throw std::runtime_error("HIP Error"); }

typedef double ZC;


// This kernel represents the "Rewrite" phase after the transfer.
// It places the incoming data into the correct bit-permuted location.
__global__ void permutation_kernel(ZC* local_buffer, ZC* incoming_data, size_t half_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < half_size) {
        // Example: Simple swap logic or interleaved rewrite
        // In a real simulator, you would use bit-masking to place 'incoming_data[tid]'
        // into the specific index required by your new bit-mapping.
        local_buffer[tid] = incoming_data[tid] * 0.5f; // Dummy math for verification
    }
}

// export HSA_FORCE_FINE_GRAINED_PCIE=1



std::vector<int> set_peers(std::vector<int> permutation) {

  std::vector<int>  R;
  int n_gpu;
  CHECK_HIP(hipGetDeviceCount(&n_gpu));

  for (int i=0; i<n_gpu; i++) {
    R.push_back(i);
  }
  
  
  // bit permutations have only 2 GPUs connections 
  for (int i=0; i<n_gpu; i++) {
    printf("i  %i p[%d]:%d \n",i,i,permutation[i]);
    
    if (permutation[i] ==i) continue;

    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipDeviceEnablePeerAccess(permutation[i], 0));
  }
  
  return R;
}

struct buffer {
  int gpu;
  // State
  ZC    *state=0;
  size_t s_bytes=0;

  // relative pointer where to read in the state
  ZC    *origin=0;

  // temporary buffer
  ZC    *temp=0;
  size_t t_bytes=0;

  void free() {
    if (state) { CHECK_HIP(hipFree(state)); state=0;} 
    if (temp) { CHECK_HIP(hipFree(temp)); temp=0;} 
  };

  void print() {
    printf(" Buffer at GPU %d State %p Size %ld Origin %p AND Temp %p  Size %ld\n",
	   gpu, (void*)state,s_bytes, (void*)origin, (void*)temp,  t_bytes);
  }
  
};

struct connection  {
  struct buffer &A;
  struct buffer &B;
  hipStream_t   s=0;

  void free() {
    A.free();
    B.free();
    if (s) { CHECK_HIP(hipStreamDestroy(s)); s = 0; }
  };
  void print() {
    printf("Source : ");
    A.print();
    printf("Dest   : ");
    B.print();
    printf("Stream Handle Value: %p\n", (void*)s);
  }

};


void set_temp(std::vector<int> gpus,
	 std::vector<int> permutation,
	 std::vector<struct buffer> &Bs,
	 std::vector<struct connection> &Cs,
	 size_t state_bytes,size_t temp_bytes  ) {

  
  for (int gpu : gpus) {
    ZC *d_state=0, *d_temp=0; 

    printf(" GPU %d alloc %zu \n", gpu, state_bytes);
    CHECK_HIP(hipSetDevice(gpu));
    CHECK_HIP(hipMalloc(&d_state, state_bytes));
    
    if (permutation[gpu] !=gpu ) {
      printf("\t GPU %d temp %zu \n", gpu, state_bytes);
      CHECK_HIP(hipMalloc(&d_temp, temp_bytes));
    } else temp_bytes=0;

    struct buffer base{gpu,d_state,state_bytes,d_state+(temp_bytes)/sizeof(ZC),d_temp,temp_bytes};
    printf("Buffer \n");
    base.print();
    Bs.push_back(base);
  }
  
  for (int gpu : gpus) {

    if (permutation[gpu] ==gpu ) continue;
    struct connection ONE{Bs[gpu],Bs[permutation[gpu]]}; 

    
    CHECK_HIP(hipSetDevice(gpu));CHECK_HIP(hipStreamCreate(&ONE.s));
    printf("Connection \n");
    

    ONE.print();
    

    Cs.push_back(ONE);
  }
  printf("buffers \n");
  for (auto &e : Bs) e.print();
  printf("connections \n");
  for (auto &e : Cs) e.print();
  

}


void run_shuffle_test(int BITS, std::vector<int> permutation) {
  
  size_t full_state_size = 1ULL << BITS; // Total BITS qubits
  size_t half_elements = full_state_size / 2;
  size_t bytes = half_elements * sizeof(ZC);

  printf("shuffle peers\n");
  std::vector<int> GPUs =  set_peers(permutation);

  std::vector<struct buffer> Bs;
  std::vector<struct connection> data;

  printf("shuffle set\n");
  set_temp(GPUs, permutation, Bs, data, 2*bytes,bytes); 
    

  auto start_ = std::chrono::high_resolution_clock::now();
  
  
  for (struct connection  &C : data) {
    printf("ASYNC communication \n");
    C.print();
    // ASYNC CROSS-GPU TRANSFER
    // GPU 0 sends its data to GPU 1's buffer
    CHECK_HIP(hipMemcpyPeerAsync(C.B.temp,   C.B.gpu,
				 C.A.origin, C.A.gpu,
				 C.A.t_bytes,
				 C.s));
    

  }
  
  // 4. The Rewrite Phase
  // Launch kernels on both GPUs to integrate the new data into the local state
  int threads = 256;
  size_t blocks = (half_elements + threads - 1) / threads;

  for (struct connection  &C : data) { 
    CHECK_HIP(hipSetDevice(C.A.gpu));
    permutation_kernel<<<blocks, threads, 0, C.s>>>(C.A.origin, C.A.temp, C.A.t_bytes/sizeof(ZC));

  }
  for (struct connection  &A : data) { 
    CHECK_HIP(hipStreamSynchronize(A.s));
  }
  
  
  auto end_ = std::chrono::high_resolution_clock::now();
  

  // 3. Calculate duration (e.g., in microseconds)
  auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>((end_ - start_));
  double time =  duration_.count()/1000000000.0;
  std::cout << " TBS: " << sizeof(ZC)*(1ULL << BITS)/time/1000000000000 << std::endl;
  
    
  std::cout << "Successfully performed HBM-to-HBM Bit Permutation Layer.\n";

  // Cleanup
  for (struct connection  A : data) { 
    A.free();
  }
  for (struct buffer  A : Bs) { 
    A.free();
  }
}

int main(int argc, char* argv[]) {
  // Method 1: Using atoi (C-style, simpler but less robust)
  int BITS  = (argc>1)? std::atoi(argv[1]):20;

  std::vector<int> permutation{0,1,3,2,4};
  
  try {
    run_shuffle_test(BITS,permutation);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return 0;
}
