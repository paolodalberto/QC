#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>



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
 * Kernel to permute columns of a matrix stored in column-major order.
 */
__global__ void permuteColumns(int N, int M, const double* d_input, 
                               const int* d_perm_indices, double* d_output) {
    
    int new_col_idx = blockIdx.x; 

    if (new_col_idx < M) {
        int old_col_idx = d_perm_indices[new_col_idx];
        
        if (threadIdx.x == 0) { 
            const double* src = d_input + old_col_idx * N;
            double* dst = d_output + new_col_idx * N;
            
            for(int i = 0; i < N; ++i) {
                dst[i] = src[i];
            }
        }
    }
}




/**
 * @brief Generates a permutation index array on the GPU by sorting keys already on the GPU.
 *
 * NOTE: The input d_sort_keys_raw will be overwritten (sorted) by this function.
 * 
 * @param M The number of columns/elements to sort.
 * @param d_sort_keys_raw Raw device pointer to the keys array on the GPU (input/output).
 * @param d_perm_indices_raw Raw device pointer for the output permutation indices (output).
 */
void generatePermutationIndicesGPU(int M, double* d_sort_keys_raw, int* d_perm_indices_raw) {
    // 1. Wrap raw device pointers in thrust::device_ptr iterators
    thrust::device_ptr<double> d_keys_begin(d_sort_keys_raw);
    thrust::device_ptr<double> d_keys_end = d_keys_begin + M;
    thrust::device_ptr<int> d_indices_begin(d_perm_indices_raw);

    // 2. Fill the output index buffer sequentially on the GPU
    // This happens entirely on the device.
    thrust::sequence(thrust::device, d_indices_begin, d_indices_begin + M);

    // 3. Perform a GPU-accelerated sort-by-key operation
    // Both input iterators are device iterators, so the operation stays on the GPU.
    thrust::sort_by_key(d_keys_begin, d_keys_end, d_indices_begin);

    // The d_perm_indices_raw pointer now holds the necessary permutation map on the GPU.
    // No hipMemcpy calls here.
}

/**** 
 * this is what we are going to use as GPU sortoarg function 
 *
 */
void sortarg(int N, // number of rows 
	     int M, // number of keys  or columns 
	     double *d_A, // source matrix  N rows and M columns we swap the columns
	     double *d_B, // permuted destination matrix 
	     double *d_sort_keys, // sorting keys
	     int* d_perm_indices // permutation 
	     ) {
	     

  // 5. Generate the permutation indices entirely on the GPU
  // Note: This operation OVERWRITES the d_sort_keys buffer with sorted keys!
  generatePermutationIndicesGPU(M, d_sort_keys, d_perm_indices);

  // 6. Launch the permutation kernel using the device-side indices
  const unsigned int blocks = M;
  const unsigned int threadsPerBlock = 1; 
  
  hipLaunchKernelGGL(permuteColumns, dim3(blocks), dim3(threadsPerBlock), 0, 0,
		     N, M, d_A, d_perm_indices, d_B);
  // d_A (input) and d_B (output) are also entirely on the GPU.

}

#ifdef SORTMAIN
int main() {
    int N = 3; // rows
    int M = 3; // columns
    size_t matrix_size_bytes = N * M * sizeof(double);
    
    // 1. Define the original host matrix
    std::vector<double> h_A = {
        10.0, 2.0, 3.0, // Col 0
        40.0, 5.0, 6.0, // Col 1
        20.0, 8.0, 9.0  // Col 2
    };

    // 2. Define the values to sort by (first row keys: 10.0, 40.0, 20.0)
    std::vector<double> h_sort_keys = { h_A[0], h_A[N], h_A[2*N] };
    
    for (double e : h_sort_keys)  std::cout << e << " \n";
    
    // 3. Allocate all necessary device memory
    double *d_A, *d_B, *d_sort_keys;
    int* d_perm_indices;
    HIP_CHECK(hipMalloc(&d_A, matrix_size_bytes));
    HIP_CHECK(hipMalloc(&d_B, matrix_size_bytes)); // Output matrix
    HIP_CHECK(hipMalloc(&d_sort_keys, M * sizeof(double))); // Keys on GPU
    HIP_CHECK(hipMalloc(&d_perm_indices, M * sizeof(int))); // Permutation map on GPU

    // 4. Copy initial data to the device (original matrix and sort keys)
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), matrix_size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sort_keys, h_sort_keys.data(), M * sizeof(double), hipMemcpyHostToDevice));

    sortarg(N, M, d_A, d_B, d_sort_keys, d_perm_indices);
    
    
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // 7. Copy ONLY the final results back to the host for verification
    std::vector<double> h_B(N * M);
    HIP_CHECK(hipMemcpy(h_B.data(), d_B, matrix_size_bytes, hipMemcpyDeviceToHost));
    
    // (Verification/Output code remains the same) ...
    std::cout << "Original Matrix (columns are not sorted):" << std::endl;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            std::cout << h_A[j * N + i] << "\t";
        }
        std::cout << std::endl;
    } 
    std::cout << "Sorted Matrix :" << std::endl;
   // ... (rest of the printout) ...
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            std::cout << h_B[j * N + i] << "\t";
        }
        std::cout << std::endl;
    }

    // 8. Cleanup all device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_sort_keys));
    HIP_CHECK(hipFree(d_perm_indices));

    return 0;
}
#endif
