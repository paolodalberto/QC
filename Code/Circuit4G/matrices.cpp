
#ifndef TYPE_OPERAND
#define TYPE_OPERAND 4 
#endif
#include "matrices.h"



__global__ void permutation_kernel(ZC* local_buffer, ZC* incoming_data, size_t half_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < half_size) {
        // Example: Simple swap logic or interleaved rewrite
        // In a real simulator, you would use bit-masking to place 'incoming_data[tid]'
        // into the specific index required by your new bit-mapping.
        local_buffer[tid] = incoming_data[tid]; 
    }
}


template <> 
void distributed_state<ZC>::set_peers() {
    
  int n_gpu;
  CHECK_HIP_ERROR(hipGetDeviceCount(&n_gpu));
  
  for (int i=0; i<n_gpu-1; i++) {
    gpus.push_back(i);
  }
};

template <> 
void distributed_state<ZC>::set_temp(std::vector<int> gpus,
				 std::vector<int> permutation,
				 std::vector<struct buffer> &Bs,
				 std::vector<struct connection> &Cs,
				 size_t state_bytes,size_t temp_bytes  ) {
  
  for (int gpu : gpus) {
    ZC *d_state=0, *d_temp=0; 
    
    printf(" GPU %d alloc %zu \n", gpu, state_bytes);
    
    if (permutation[gpu] !=gpu ) {
      CHECK_HIP_ERROR(hipSetDevice(gpu));
      printf("\t GPU %d temp %zu \n", gpu, state_bytes);
      CHECK_HIP_ERROR(hipMalloc(&d_temp, temp_bytes));
    } //else temp_bytes=0;
    
    struct buffer base{gpu,G[gpu].d_matrix,state_bytes,G[gpu].d_matrix+(temp_bytes)/sizeof(ZC),d_temp,temp_bytes};
    printf("Buffer \n");
    base.print();
      Bs.push_back(base);
  }
  
  for (int gpu : gpus) {
    
    if (permutation[gpu] ==gpu ) continue;
    struct connection ONE{Bs[gpu],Bs[permutation[gpu]]}; 
    CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&ONE.can_access, gpu, permutation[gpu]));
    CHECK_HIP_ERROR(hipSetDevice(gpu));
    if (ONE.can_access) {
      
      CHECK_HIP_ERROR(hipDeviceEnablePeerAccess(permutation[gpu], 0));
    }
    
    CHECK_HIP_ERROR(hipStreamCreate(&ONE.s));
    printf("Connection \n");
    
    
    ONE.print();
    
    
    Cs.push_back(ONE);
  }
  printf("buffers \n");
  for (auto &e : Bs) e.print();
  printf("connections \n");
  for (auto &e : Cs) e.print();
  
  
};

template <> 
void  distributed_state<ZC>::init() {
  size = 1ULL << bits;
  set_peers();
  for (int i=0; i<gpus.size(); i++) {
    Matrix M{size,1,size,1};
    M.gpu = gpus[i];
    M.alloc(true,true);
    M.print();
    G.push_back(M);
  }
  
  
}

template <> 
void  distributed_state<ZC>::free() {

  for (int i=0; i<G.size(); i++) {
    G[i].print(true);
    G[i].free();
  }
  G.clear();
  
}



template <> 
void  distributed_state<ZC>::run_shuffle_test(std::vector<int> permutation) {
  
  size_t full_state_size = G[0].size(); // Total BITS qubits
  size_t half_elements = full_state_size / 2;
  size_t bytes = half_elements * sizeof(ZC);
  
  
  ZC *h_stage;
  // Pinned memory is required for ~3x bandwidth and safe staging
  CHECK_HIP_ERROR(hipHostMalloc(&h_stage, bytes)); 
  

  std::vector<int> GPUs =  gpus;
  
  std::vector<struct buffer> Bs;
  std::vector<struct connection> data;
  
  printf("shuffle set\n");
  set_temp(GPUs, permutation, Bs, data, 2*bytes,bytes); 
  
  
  auto start_ = std::chrono::high_resolution_clock::now();
  
  
  for (struct connection  &C : data) {
    // ASYNC CROSS-GPU TRANSFER
    // GPU 0 sends its data to GPU 1's buffer
    //    CHECK_HIP_ERROR_ERROR(hipSetDevice(C.A.gpu)); 
    if (C.can_access) {
      printf("ASYNC communication \n");
      C.print();
      
      CHECK_HIP_ERROR(hipMemcpyPeerAsync(C.B.temp,   C.B.gpu,
				   C.A.origin, C.A.gpu,
				   C.A.t_bytes,
				   C.s
				   ));
    }
  }
  for (struct connection  &C : data) {
    // ASYNC CROSS-GPU TRANSFER
    // GPU 0 sends its data to GPU 1's buffer
    //    CHECK_HIP_ERROR(hipSetDevice(C.A.gpu)); 
    if (!C.can_access) { 
      printf("SYNC communication \n");
      C.print();
      // Step A: Source GPU -> Host
      CHECK_HIP_ERROR(hipSetDevice(C.A.gpu));
      CHECK_HIP_ERROR(hipMemcpyAsync(h_stage, C.A.origin, C.A.t_bytes, hipMemcpyDeviceToHost, C.s));
      
      // Step B: Ensure the D2H copy is finished before Host -> Destination GPU starts
      CHECK_HIP_ERROR(hipStreamSynchronize(C.s)); 
      
      // Step C: Host -> Destination GPU
      CHECK_HIP_ERROR(hipSetDevice(C.B.gpu));
      CHECK_HIP_ERROR(hipMemcpyAsync(C.B.temp, h_stage, C.A.t_bytes, hipMemcpyHostToDevice, C.s));
      CHECK_HIP_ERROR(hipStreamSynchronize(C.s)); 
    }
    
  }
  
  // 4. The Rewrite Phase
    // Launch kernels on both GPUs to integrate the new data into the local state
  int threads = 256;
  size_t blocks = (half_elements + threads - 1) / threads;

  for (struct connection  &C : data) { 
    CHECK_HIP_ERROR(hipSetDevice(C.B.gpu));
    permutation_kernel<<<blocks, threads, 0, C.s>>>(C.B.origin, C.B.temp, C.B.t_bytes/sizeof(ZC));
    
  }
  for (struct connection  &A : data) { 
    CHECK_HIP_ERROR(hipStreamSynchronize(A.s));
  }
  
  
  auto end_ = std::chrono::high_resolution_clock::now();
  
  
  // 3. Calculate duration (e.g., in microseconds)
  auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>((end_ - start_));
  double time =  duration_.count()/1000000000.0;
  std::cout << " TBS: " << sizeof(ZC)*(full_state_size)/time/1000000000000 << std::endl;
  
  
  std::cout << "Successfully performed HBM-to-HBM Bit Permutation Layer.\n";
  
  // Cleanup
  for (struct connection  A : data) { 
    A.free();
  }
  for (struct buffer  A : Bs) { 
    A.free();
  }
};


