#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Macro for HIP error checking
#define CHECK_HIP(expression)                                           \
{                                                                       \
    hipError_t status = (expression);                                   \
    if (status != hipSuccess) {                                         \
        throw std::runtime_error("HIP Error: " +                        \
            std::string(hipGetErrorString(status)) + " at " + __FILE__); \
    }                                                                   \
}

// Macro for rocBLAS error checking
#define CHECK_ROCBLAS(expression)                                       \
{                                                                       \
    rocblas_status status = (expression);                               \
    if (status != rocblas_status_success) {                             \
        throw std::runtime_error("rocBLAS Error at " + std::string(__FILE__)); \
    }                                                                   \
}

class ROCblasPool {
    using TaskFunc = std::function<void(rocblas_handle, hipStream_t)>;
    struct Worker {
        std::thread thread;
        std::queue<TaskFunc> tasks;
        std::mutex mtx;
        std::condition_variable cv;
    };

    std::vector<std::unique_ptr<Worker>> workers;
    bool stop = false;

public:
    ROCblasPool(int num_devices) {
        for (int i = 0; i < num_devices; ++i) {
            auto worker = std::make_unique<Worker>();
            auto* w_ptr = worker.get();
            workers.push_back(std::move(worker));

            w_ptr->thread = std::thread([this, i, w_ptr]() {
                try {
                    // 1. Set Device & Context
                    CHECK_HIP(hipSetDevice(i));

                    rocblas_handle handle;
                    hipStream_t stream;

                    // 2. Initialize Resources
                    CHECK_HIP(hipStreamCreate(&stream));
                    CHECK_ROCBLAS(rocblas_create_handle(&handle));
                    CHECK_ROCBLAS(rocblas_set_stream(handle, stream));
                    
                    // 3. Warm up kernels
                    rocblas_initialize();

                    while (true) {
                        TaskFunc task;
                        {
                            std::unique_lock<std::mutex> lock(w_ptr->mtx);
                            w_ptr->cv.wait(lock, [this, w_ptr] { 
                                return stop || !w_ptr->tasks.empty(); 
                            });
                            if (stop && w_ptr->tasks.empty()) break;
                            task = std::move(w_ptr->tasks.front());
                            w_ptr->tasks.pop();
                        }
                        task(handle, stream);
                    }

                    CHECK_ROCBLAS(rocblas_destroy_handle(handle));
                    CHECK_HIP(hipStreamDestroy(stream));
                } catch (const std::exception& e) {
                    std::cerr << "Worker " << i << " failed: " << e.what() << std::endl;
                }
            });
        }
    }

    void enqueue(int device_id, TaskFunc f) {
        {
            std::lock_guard<std::mutex> lock(workers[device_id]->mtx);
            workers[device_id]->tasks.push(std::move(f));
        }
        workers[device_id]->cv.notify_one();
    }

    ~ROCblasPool() {
        stop = true;
        for (auto& w : workers) {
            w->cv.notify_all();
            if (w->thread.joinable()) w->thread.join();
        }
    }
};

// --- Test Case: Split Vector GEMM ---
int main() {
    int num_devices = 0;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (num_devices < 2) {
        std::cout << "Detected " << num_devices << " GPU(s). Need 2 for this test." << std::endl;
        return 0;
    }

    ROCblasPool pool(num_devices);
    const int M = 512, N = 512, K = 512;

    for (int i = 0; i < 2; ++i) {
        pool.enqueue(i, [i, M, N, K](rocblas_handle h, hipStream_t s) {
            float *d_A, *d_B, *d_C;
            float alpha = 1.0f, beta = 0.0f;

            // Allocation with Check
            CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(float)));
            CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(float)));
            CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(float)));

            // Execute GEMM
            CHECK_ROCBLAS(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                                        M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));

            // Synchronize and Clean
            CHECK_HIP(hipStreamSynchronize(s));
            std::cout << "GPU " << i << " successfully completed GEMM." << std::endl;

            CHECK_HIP(hipFree(d_A));
            CHECK_HIP(hipFree(d_B));
            CHECK_HIP(hipFree(d_C));
        });
    }

    return 0;
}
