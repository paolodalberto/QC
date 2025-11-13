#include <rocprim/rocprim.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>


#define HIP_CHECK(call) {                                    \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
        std::cerr << "HIP error in " << __FILE__ << ":"     \
                  << __LINE__ << " : " << hipGetErrorString(err) \
                  << std::endl;                              \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}

struct custom_type {
  double value;
  double vector[2];
};

// Custom comparator functor for descending order
struct DescendingCompare {
    __device__ bool operator()(const custom_type& a, const custom_type& b) const {
        return a.value < b.value; // Use 'a > b' for descending order
    }
};




void romPrimSortExampleWithChecks() {
  
    std::vector<custom_type> input_data = {
      { 4.5f, {10.0, 10.0}},
      { 4.8f, {10.0, 10.0}},
      { 4.2f, {10.0, 10.0}},
      { 4.1f, {10.0, 10.0}},
      { 5.5f, {10.0, 10.0}},
      { 1.5f, {10.0, 10.0}},
      {5.0f,  {10.0, 10.0}}
    };
    size_t input_size = input_data.size();
  
    custom_type* d_input;
    custom_type* d_output;
    
    // Allocate device memory and check for errors
    HIP_CHECK(hipMalloc(&d_input, input_size * sizeof(custom_type)));
    HIP_CHECK(hipMalloc(&d_output, input_size * sizeof(custom_type)));

    // Copy data and check for errors
    HIP_CHECK(hipMemcpy(d_input, input_data.data(), input_size * sizeof(custom_type), hipMemcpyHostToDevice));

    size_t temp_storage_size_bytes;
    void* d_temp_storage = nullptr;

    // First call to get required size for temporary storage and check error
    HIP_CHECK(rocprim::merge_sort(
        d_temp_storage, 
        temp_storage_size_bytes, 
        d_input, 
        d_output, 
        input_size,
	DescendingCompare{} 
    ));

    // Allocate temporary storage on the device and check error
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    // Perform the device sort and check error
    HIP_CHECK(rocprim::merge_sort(
        d_temp_storage, 
        temp_storage_size_bytes, 
        d_input, 
        d_output, 
        input_size,
	DescendingCompare{} 
    ));

    // Copy sorted output back to host and check error
    std::vector<custom_type> output_data(input_size);
    HIP_CHECK(hipMemcpy(output_data.data(), d_output, input_size * sizeof(custom_type), hipMemcpyDeviceToHost));

    std::cout << "Sorted data: ";
    for (custom_type val : output_data) {
        std::cout << val.value << " ";
    }
    std::cout << std::endl;

    // Free device memory and check for errors
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}


int main()
{
  romPrimSortExampleWithChecks();
  return 0; 
}
