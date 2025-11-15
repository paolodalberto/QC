#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

// Helper macro for checking HIP errors
#define HIP_CHECK(call) {                                    \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
        std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                  << " : " << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}

/**
 * Kernel for element-wise multiplication with broadcasting.
 * Assumes column-major storage for the matrix.
 * Vector 'd_a' is broadcast across the columns of matrix 'd_b'.
 */
__global__ void elementwise_mult_broadcast(int N, int M, 
                                           const double* d_a, // Vector [M elements]
                                           const double* d_b, // Matrix [N*M elements]
                                           double* d_c)       // Output Matrix [N*M elements]
{
    // Calculate global thread ID in X and Y dimensions (row and column)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        // Index calculation for column-major storage: index = row + col * N
        
        double vector_val = d_a[col];      // The value to broadcast for this column
        double matrix_val = d_b[row + col * N]; // The matrix element
        
        d_c[row + col * N] = matrix_val * vector_val;
    }
}

int main() {
    int N = 3; // rows
    int M = 3; // columns
    size_t matrix_size_bytes = N * M * sizeof(double);
    size_t vector_size_bytes = M * sizeof(double);

    // 1. Define host data
    // Vector 'a' = [10.0, 20.0, 30.0]
    std::vector<double> h_a = {10.0, 20.0, 30.0}; 

    // Matrix 'b' (Column-Major Order)
    // 1.0  4.0  7.0
    // 2.0  5.0  8.0
    // 3.0  6.0  9.0
    std::vector<double> h_b = {
        1.0, 2.0, 3.0, // Col 0
        4.0, 5.0, 6.0, // Col 1
        7.0, 8.0, 9.0  // Col 2
    };
    std::vector<double> h_c(N * M); // Host output buffer

    // Expected Output Matrix 'c': Col[i] * h_a[i]
    // 10*1.0  20*4.0  30*7.0   =>  10.0  80.0  210.0
    // 10*2.0  20*5.0  30*8.0   =>  20.0 100.0  240.0
    // 10*3.0  20*6.0  30*9.0   =>  30.0 120.0  270.0

    // 2. Allocate device memory
    double *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, vector_size_bytes));
    HIP_CHECK(hipMalloc(&d_b, matrix_size_bytes));
    HIP_CHECK(hipMalloc(&d_c, matrix_size_bytes));

    // 3. Copy data to the device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), vector_size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), matrix_size_bytes, hipMemcpyHostToDevice));

    // 4. Launch the kernel
    // We launch a 2D grid of threads (N rows * M columns is a safe size)
    dim3 blocks( (N + 15) / 16, (M + 15) / 16 ); // Example block size calculation
    dim3 threadsPerBlock(16, 16); 

    elementwise_mult_broadcast<<<blocks, threadsPerBlock>>>(
        N, M, d_a, d_b, d_c
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize()); 

    // 5. Copy the result back to the host
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, matrix_size_bytes, hipMemcpyDeviceToHost));

    // 6. Verification and Output
    std::cout << "Input Vector a: ";
    for(double val : h_a) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "\nResult Matrix C (display):" << std::endl;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            // Accessing element at row i, col j in column-major storage
            std::cout << h_c[j * N + i] << "\t"; 
        }
        std::cout << std::endl;
    }

    // 7. Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return 0;
}
