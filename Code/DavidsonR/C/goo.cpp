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

struct Value_Index {
  double value;
  int    index;
};

// Custom comparator functor for descending order
struct CompareFunction {
  __device__ bool operator()(const Value_Index& a, const Value_Index& b) const {
    return a.value < b.value; // Use 'a > b' for descending order
  }
};







void romPrimSortExampleWithChecks() {
  // input and initialization
  std::vector<Value_Index> input_data = {
    { 4.5f, 0},
    { 4.8f, 1},
    { 4.2f, 2},
    { 4.1f, 3},
    { 5.5f, 4},
    { 1.5f, 5},
    { 5.0f, 6}
  };
  size_t input_size = input_data.size();
  std::vector<Value_Index> output_data(input_size);
  

    
  Value_Index* d_input;
  Value_Index* d_output;
  size_t temp_storage_size_bytes = input_size * sizeof(Value_Index);
  void* d_temp_storage = nullptr;


    
  // Allocate device memory and check for errors
  HIP_CHECK(hipMalloc(&d_input, input_size * sizeof(Value_Index)));
  HIP_CHECK(hipMalloc(&d_output, input_size * sizeof(Value_Index)));

  // Allocate temporary storage on the device and check error
  HIP_CHECK(hipMalloc(&d_temp_storage,  input_size * sizeof(Value_Index)));
  
    
  // Copy data and check for errors
  HIP_CHECK(hipMemcpy(d_input, input_data.data(), input_size * sizeof(Value_Index), hipMemcpyHostToDevice));
  

  // First call to get required size for temporary storage and check error
  /* We allocate space already 
    HIP_CHECK(rocprim::merge_sort(
        d_temp_storage, 
        temp_storage_size_bytes, 
        d_input, 
        d_output, 
        input_size,
	DescendingCompare{} 
    ));
  */


  // Perform the device sort and check error
  HIP_CHECK(
	    rocprim::merge_sort(
				d_temp_storage, 
				temp_storage_size_bytes, 
				d_input, 
				d_output, 
				input_size,
				CompareFunction{} 
				)
	    );

  // Copy sorted output back to host and check error
    
  HIP_CHECK(hipMemcpy(output_data.data(), d_output, input_size * sizeof(Value_Index), hipMemcpyDeviceToHost));
  
  std::cout << "Sorted data: ";
  for (Value_Index val : output_data) {
    std::cout << val.value << " " << val.index << "; ";
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
