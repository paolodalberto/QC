#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define CHECK_HIP(expr) { hipError_t status = (expr); if(status != hipSuccess) throw std::runtime_error("HIP Error"); }

// This kernel represents the "Rewrite" phase after the transfer.
// It places the incoming data into the correct bit-permuted location.
__global__ void permutation_kernel(float* local_buffer, float* incoming_data, size_t half_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < half_size) {
        // Example: Simple swap logic or interleaved rewrite
        // In a real simulator, you would use bit-masking to place 'incoming_data[tid]'
        // into the specific index required by your new bit-mapping.
        local_buffer[tid] = incoming_data[tid] * 0.5f; // Dummy math for verification
    }
}

// export HSA_FORCE_FINE_GRAINED_PCIE=1


void run_shuffle_test() {
    int n_gpu;
    CHECK_HIP(hipGetDeviceCount(&n_gpu));
    if (n_gpu < 2) return;

    // 1. Enable Peer-to-Peer Access
    // This allows GPU 0 to read/write GPU 1's memory directly via Infinity Fabric/PCIe
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipDeviceEnablePeerAccess(1, 0));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipDeviceEnablePeerAccess(0, 0));

    // 2. Prepare Memory
    size_t full_state_size = 1ULL << 14; // Total 14 qubits
    size_t half_elements = full_state_size / 2;
    size_t bytes = half_elements * sizeof(float);

    float *d_state0, *d_state1, *d_remote_buffer0, *d_remote_buffer1;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&d_state0, bytes));
    CHECK_HIP(hipMalloc(&d_remote_buffer0, bytes)); // Buffer to receive from GPU 1

    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMalloc(&d_state1, bytes));
    CHECK_HIP(hipMalloc(&d_remote_buffer1, bytes)); // Buffer to receive from GPU 0

    // 3. The Shuffle Task (The "Barrier" Layer)
    hipStream_t s0, s1;
    CHECK_HIP(hipSetDevice(0)); CHECK_HIP(hipStreamCreate(&s0));
    CHECK_HIP(hipSetDevice(1)); CHECK_HIP(hipStreamCreate(&s1));

    // ASYNC CROSS-GPU TRANSFER
    // GPU 0 sends its data to GPU 1's buffer
    CHECK_HIP(hipMemcpyPeerAsync(d_remote_buffer1, 1, d_state0, 0, bytes, s0));
    
    // GPU 1 sends its data to GPU 0's buffer
    CHECK_HIP(hipMemcpyPeerAsync(d_remote_buffer0, 0, d_state1, 1, bytes, s1));

    // 4. The Rewrite Phase
    // Launch kernels on both GPUs to integrate the new data into the local state
    int threads = 256;
    int blocks = (half_elements + threads - 1) / threads;

    CHECK_HIP(hipSetDevice(0));
    permutation_kernel<<<blocks, threads, 0, s0>>>(d_state0, d_remote_buffer0, half_elements);

    CHECK_HIP(hipSetDevice(1));
    permutation_kernel<<<blocks, threads, 0, s1>>>(d_state1, d_remote_buffer1, half_elements);

    // Sync everything
    CHECK_HIP(hipStreamSynchronize(s0));
    CHECK_HIP(hipStreamSynchronize(s1));

    std::cout << "Successfully performed HBM-to-HBM Bit Permutation Layer.\n";

    // Cleanup
    CHECK_HIP(hipFree(d_state0)); CHECK_HIP(hipFree(d_remote_buffer0));
    CHECK_HIP(hipFree(d_state1)); CHECK_HIP(hipFree(d_remote_buffer1));
}

int main() {
    try {
        run_shuffle_test();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
