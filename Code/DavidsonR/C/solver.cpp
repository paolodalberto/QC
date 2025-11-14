#include <iostream>
#include <vector>
#include <rocblas/rocblas.h>   // Change from <rocblas.h>
#include <rocsolver/rocsolver.h> // Change from <rocsolver.h>


// Helper function for error checking (simplified)
void check_hip_error(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void check_rocblas_error(rocblas_status status) {
    if (status != rocblas_status_success) {
        std::cerr << "rocBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}



#define CHECK_ROCSOLVER(call) {                             \
    rocblas_status status = call;                           \
    if (status != rocblas_status_success) {				\
      std::cerr << "rocSOLVER error at " << __FILE__ << ":" << __LINE__ \
		<< std::endl;						\
      exit(EXIT_FAILURE);						\
    }									\
  }

int main() {
    // Matrix size
    int n = 3;
    // Leading dimension of A (for this example, lda = n)
    int lda = n;
    // Buffer size for workspace (optimal size can be queried, here we provide a reasonable default)
    int lwork = 2*n*n; // This might need adjustment based on rocSOLVER documentation

    // Host side data
    // Example symmetric matrix A (stored column-major)
    std::vector<double> A_host = {
        1.0, 2.0, 3.0,
        2.0, 4.0, 5.0,
        3.0, 5.0, 6.0
    };
    std::vector<double> W_host(n); // Eigenvalues will be stored here
    int info_host = 0; // Error indicator

    // Device side pointers
    double *A_device, *W_device, *work_device;
    int *info_device;
    rocblas_handle handle;

    // 1. Initialize rocBLAS handle
    check_rocblas_error(rocblas_create_handle(&handle));

    // 2. Allocate device memory
    check_hip_error(hipMalloc(&A_device, sizeof(double) * n * n));
    check_hip_error(hipMalloc(&W_device, sizeof(double) * n));
    check_hip_error(hipMalloc(&work_device, sizeof(double) * n));
    check_hip_error(hipMalloc(&info_device, sizeof(int)));

    // 3. Copy matrix A to device
    check_hip_error(hipMemcpy(A_device, A_host.data(), sizeof(double) * n * n, hipMemcpyHostToDevice));

    // 4. Call rocsolver_dsyev 
    // Arguments are now directly the enum values (rocblas_evect_full, rocblas_fill_upper)
    // instead of the 'V'/'U' characters.
    check_rocblas_error(
			rocsolver_dsyev(
					handle,
					rocblas_evect_original, //rocblas_evect_full, // Jobz: use the enum value
					rocblas_fill_upper, // Uplo: use the enum value
					n,                  
					A_device,           
					n,                  
					W_device,
					work_device,
					info_device         
					));
    /*
    rocblas_status rocsolver_dsyev(
				   rocblas_handle handle,
				   const rocblas_evect evect,
				   const rocblas_fill uplo,
				   const rocblas_int n,
				   double *A,
				   const rocblas_int lda,
				   double *D,
				   double *work_device,
				   rocblas_int *info)
    */

    // 5. Copy results (eigenvalues, updated A matrix containing eigenvectors, and info) back to host
    check_hip_error(hipMemcpy(W_host.data(), W_device, sizeof(double) * n, hipMemcpyDeviceToHost));
    check_hip_error(hipMemcpy(A_host.data(), A_device, sizeof(double) * n * n, hipMemcpyDeviceToHost));
    check_hip_error(hipMemcpy(&info_host, info_device, sizeof(int), hipMemcpyDeviceToHost));

    // 6. Check for convergence issues and print results
    if (info_host == 0) {
        std::cout << "Eigenvalues: ";
        for (int i = 0; i < n; ++i) {
            std::cout << W_host[i] << (i == n - 1 ? "" : ", ");
        }
        std::cout << std::endl;

        std::cout << "Eigenvectors (columns of the matrix):" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << A_host[j * n + i] << " "; // Accessing column-major data
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "The algorithm failed to converge. info = " << info_host << std::endl;
    }

    // 7. Deallocate memory
    check_hip_error(hipFree(A_device));
    check_hip_error(hipFree(W_device));
    check_hip_error(hipFree(work_device));
    check_hip_error(hipFree(info_device));
    check_rocblas_error(rocblas_destroy_handle(handle));

    return 0;
}
