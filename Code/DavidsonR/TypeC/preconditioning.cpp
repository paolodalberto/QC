
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::iota
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include "davidson.h"



// --- HIP Kernel for Batched Residual Calculation (R = C - lambda*X) ---
// This handles the R(i,k) = (H*X)(i,k) - lambda(k) * X(i,k) step
extern "C" __global__ void compute_residuals_kernel(
    const ZC* d_HX_inter,   // H * X intermediate results (M x N_EIG)
    const ZC* d_X,          // Eigenvectors X (M x N_EIG)
    const NORM_TYPE* d_eigvals,    // Eigenvalues (N_EIG vector)
    ZC* d_R,                // Output residuals R (M x N_EIG)
    const size_t M,             // Number of rows
    const size_t N_EIG)         // Number of columns/eigenvectors
{
    size_t k = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    size_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (k < N_EIG && i < M) {
        size_t index = k * M + i; // Column-major index
        d_R[index] = d_HX_inter[index] - d_eigvals[k] * d_X[index];
    }
}


// --- HIP Kernel for Batched Preconditioning ---
// Computes T(i,k) = -R(i,k) / (diag_H(i) - eig_val(k) + epsilon)
extern "C" __global__ void davidson_preconditioner_2D_kernel(
    const ZC* d_R,          // Input Residuals (M x N_EIG matrix, column major)
    const ZC* d_diag_H,     // Input Diagonal (M vector)
    ZC* d_T,                // Output Corrections (M x N_EIG matrix, column major)
    const NORM_TYPE* d_eig_vals,   // Input Eigenvalues (N_EIG vector)
    const ZC epsilon,       
    const size_t M,             
    const size_t N_EIG)         
{
    size_t k = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    size_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (k < N_EIG && i < M) {
        size_t index = k * M + i; // Column-major index
        
        // Denominator depends on the row index (i) and column index (k)
        ZC denominator = d_diag_H[i] - d_eig_vals[k] + epsilon;
        
        d_T[index] = -d_R[index] / denominator;
    }
}


// =======================================================
// PHASE 1: COMPUTE RESIDUALS d_HX = d_H*d_X
//          d_R = d_HX -  d_eig_vals*d_X
// =======================================================
void residuals(rocblas_handle handle,
	       int M, int N_EIG,
	       ZC *d_H, 
	       ZC *d_X, // Eigenvectors 
	       NORM_TYPE *d_eig_vals, // eigenvalues 
	       ZC *d_HX_inter, // intermediate result
	       ZC *d_R  // residuals
	       )  {
  

    // =======================================================
    // PHASE 1: COMPUTE RESIDUALS R = H*X - Lambda*X (Batched)
    // =======================================================
    //std::cout << "Calculating Residual Vectors R = H*X - lambda*X" << std::endl;

    // 1a. Compute d_HX_inter = H * X using rocblas_dgemm
    ZC alpha_dgemm = ALPHA;
    ZC beta_dgemm = BETA; // Overwrite temp buffer
    CHECK_ROCBLAS_STATUS(
        GEMM(
	     handle, rocblas_operation_none, rocblas_operation_none,
	     M, N_EIG, M,
	     &alpha_dgemm, d_H, M,
	     d_X, M,
	     &beta_dgemm, d_HX_inter, M // d_HX_inter now holds H*X
    ));

    // 1b. Compute R = d_HX_inter - Lambda*X using the 2D custom HIP kernel
    dim3 threadsPerBlock_R(16, 16); 
    dim3 blocksPerGrid_R(
        (N_EIG + threadsPerBlock_R.x - 1) / threadsPerBlock_R.x, 
        (M + threadsPerBlock_R.y - 1) / threadsPerBlock_R.y      
    );
    hipLaunchKernelGGL(
        compute_residuals_kernel,
        blocksPerGrid_R,
        threadsPerBlock_R,
        0, 0,
        d_HX_inter, d_X, d_eig_vals, d_R, M, N_EIG
    );
    CHECK_HIP_ERROR(hipGetLastError());
    CHECK_HIP_ERROR(hipDeviceSynchronize()); 
    // At this point, d_R correctly holds all residual vectors



}

void corrections(int M, int N_EIG,
		 ZC *d_R,      // residuals 
		 ZC *d_diag_H, // H diagonal
		 NORM_TYPE *d_eig_vals, // eigenvalues 
		 ZC *d_T,         // result
		 const ZC epsilon) { 

    // =======================================================
    // PHASE 2: COMPUTE CORRECTIONS T using the Preconditioner (Batched)
    // =======================================================
    //std::cout << "Calculating Correction Vectors T using Batched Preconditioner..." << std::endl;


    dim3 threadsPerBlock_T(16, 16);
    dim3 blocksPerGrid_T(
        (N_EIG + threadsPerBlock_T.x - 1) / threadsPerBlock_T.x,
        (M + threadsPerBlock_T.y - 1) / threadsPerBlock_T.y
    );

    // Launch the Batched Davidson Preconditioner Kernel
    hipLaunchKernelGGL(
        davidson_preconditioner_2D_kernel,
        blocksPerGrid_T,
        threadsPerBlock_T,
        0, 0,
        d_R, d_diag_H, d_T, d_eig_vals, epsilon, M, N_EIG
    );
    CHECK_HIP_ERROR(hipGetLastError());
    CHECK_HIP_ERROR(hipDeviceSynchronize()); 
    //std::cout << "Preconditioning complete. Correction matrix T is in device memory d_T." << std::endl;


}



#ifdef PREMAIN 
// --- Main Example Demonstrating the Full Cycle ---

int main() {
    const size_t M = 100;   // Matrix dimension M x M
    const size_t N_EIG = 3; // Number of eigenvectors/batches we process
    const ZC epsilon = 1e-12;

    
    // --- Host Data ---
    std::vector<ZC> h_H(M * M, 0.0);       // Full Hamiltonian (for init)
    std::vector<ZC> h_X(M * N_EIG);        // Approximate Eigenvectors (input)
    std::vector<ZC> h_diag_H(M);           // Diagonal elements (input to kernel)
    std::vector<ZC> h_eig_vals(N_EIG);     // Approximate Eigenvalues (input to kernel)
    std::vector<ZC> h_T(M * N_EIG);        // Correction vector results (output)


    // Initialize H as a diagonally dominant matrix
    for (size_t i = 0; i < M; ++i) {
        h_H[i * M + i] = (ZC)(i * 10.0 + 1.0); 
        h_diag_H[i] = h_H[i * M + i];
    }

    // Initialize N_EIG eigenvector guesses (normalized constant vectors)
    for (size_t k = 0; k < N_EIG; ++k) {
        for (size_t i = 0; i < M; ++i) {
             // Use 0.1 as initial value for simplicity in verification
            h_X[k * M + i] = 0.1; 
        }
    }
    
    // Initialize approximate eigenvalues (arbitrary but distinct values for verification)
    h_eig_vals[0] = 50.0;
    h_eig_vals[1] = 150.0;
    h_eig_vals[2] = 250.0;


    // --- Device Memory Allocation ---
    ZC *d_H, *d_X, *d_diag_H, *d_eig_vals, *d_R, *d_T, *d_HX_inter;
    CHECK_HIP_ERROR(hipMalloc(&d_H, M * M * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_X, M * N_EIG * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_diag_H, M * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_eig_vals, N_EIG * sizeof(ZC)));
    CHECK_HIP_ERROR(hipMalloc(&d_R, M * N_EIG * sizeof(ZC))); // Residuals
    CHECK_HIP_ERROR(hipMalloc(&d_T, M * N_EIG * sizeof(ZC))); // Corrections
    CHECK_HIP_ERROR(hipMalloc(&d_HX_inter, M * N_EIG * sizeof(ZC))); // H*X temp storage

    // Copy necessary data to device
    CHECK_HIP_ERROR(hipMemcpy(d_H, h_H.data(), M * M * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_X, h_X.data(), M * N_EIG * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_diag_H, h_diag_H.data(), M * sizeof(ZC), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_eig_vals, h_eig_vals.data(), N_EIG * sizeof(ZC), hipMemcpyHostToDevice));


    // --- rocBLAS Initialization ---
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // =======================================================
    // PHASE 1: COMPUTE RESIDUALS R = H*X - Lambda*X (Batched)
    // =======================================================

    residuals(handle,
	      M, N_EIG,
	      d_H,
	      d_X,d_eig_vals,
	      d_HX_inter,d_R);
	      

    
    // =======================================================
    // PHASE 2: COMPUTE CORRECTIONS T using the Preconditioner (Batched)
    // =======================================================
    corrections(M,N_EIG,
		d_R,      // residuals 
		d_diag_H, // H diagonal
		d_eig_vals, // eigenvalues 
		d_T, epsilon);


    // =======================================================
    // VERIFICATION: Copy results back to host and check
    // =======================================================
    CHECK_HIP_ERROR(hipMemcpy(h_T.data(), d_T, M * N_EIG * sizeof(ZC), hipMemcpyDeviceToHost));
    
    std::cout << "\nVerification of T[0] for all " << N_EIG << " eigenvectors:" << std::endl;
    for (size_t k = 0; k < N_EIG; ++k) {
        // Calculate expected T[0, k] value:
        // R[0, k] = (H[0,0]*X[0,k]) - lambda[k]*X[0,k]
        // T[0, k] = -R[0, k] / (H[0,0] - lambda[k] + epsilon)
        ZC expected_R_0k = (h_H[0] * h_X[k*M]) - h_eig_vals[k] * h_X[k*M];
        ZC expected_T_0k = -expected_R_0k / (h_diag_H[0] - h_eig_vals[k] + epsilon);

        std::cout << "  Eigenvector " << k << " (Lambda=" << h_eig_vals[k] << "):"
                  << " Got T[0," << k << "]: " << h_T[k*M] 
                  << ", Expected T[0," << k << "]: " << expected_T_0k << std::endl;
    }


    // --- Cleanup ---
    CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));
    CHECK_HIP_ERROR(hipFree(d_H));
    CHECK_HIP_ERROR(hipFree(d_X));
    CHECK_HIP_ERROR(hipFree(d_diag_H));
    CHECK_HIP_ERROR(hipFree(d_eig_vals));
    CHECK_HIP_ERROR(hipFree(d_R));
    CHECK_HIP_ERROR(hipFree(d_T));
    CHECK_HIP_ERROR(hipFree(d_HX_inter));

    return EXIT_SUCCESS;
}
#endif
