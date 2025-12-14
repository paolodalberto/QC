


/*********************************************************
 * The Idea is simple: The computation of a Gate G on a state S is
 * expressed by a a kronecker computation (x) as 
 *
 * (I_n (x) G (x) I_k) * S
 *
 * I_n stands for an identity matrix and n amd k mean the number of
 * bit that is the Gate applied to m bits the G is applied to bit k,
 * k+1, ... k+m-1 and the total number of bits is n + m + k and the
 * state is of size 2^(n+m+k). 
 *
 * We showed this in the Python implementation and also we show that
 * this boils down to the computation of 2^n matrix multiplicaitons
 * (strided) G over the remainder state S[0... 2^(k+m)] as a matrix of
 * size 2^m x 2^k ... so far brilliant
 *
 **************/


#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>


// we define the computation double complex 

#define TYPE_OPERAND 4 
#include "davidson.h"  // definition of matrices 
#include "circuit.h"   // definition of Gate and Circuit



int set_device(int id) {
  int deviceCount;
  CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return -1;
  }

  // Enumerate devices and their properties (optional, but good practice)
  for (int i = 0; i < deviceCount; ++i) {
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, i));
    std::cout << "Device " << i << ": " << props.name << std::endl;
  }

  // Set the desired device (e.g., device 0)
  int desiredDevice = id; 
  if (desiredDevice < deviceCount) {
    CHECK_HIP_ERROR(hipSetDevice(desiredDevice));
    std::cout << "Successfully set device to " << desiredDevice << std::endl;
  } else {
    std::cerr << "Invalid device index: " << desiredDevice << std::endl;
    return -1;
  }
  return id;
}


/*****
 * this is column major and thus we will need to compute O = I * G^t
 * but I will transpose directly G  ...
 ***/

void cpu_zgemm_batched_M(
     int M, int N, int K, ZC alpha, 
     Matrix &A,
     Matrix &B,  // B is the small one the gate one 
     ZC beta,
     Matrix &C,
     int batchCount) {
  
  cpu_zgemm_batched_b(
	   A.m, B.m, A.n, alpha, 
	   A.matrix, A.m, // I 
	   B.matrix, B.m, // G^t
	   beta,
	   C.matrix, C.m, // O
	   batchCount
		      );
  
}



extern Gate CNot;
extern const Gate Hadamard;


int main(int argc, char* argv[]) {
   // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = (argc>1)? std::atoi(argv[1]):0;
  int result =  set_device(mydevice);
  int M = 2;
  
  printf(" device: %d ;  State Bits %d \n", mydevice, M);
  
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));


  Matrix Input = {M,1,M,1};
  Input.alloc(true,true);



  // building teh circuit like we belong
  Gate H0 = Hadamard; H0.set_index(0);
  Gate H1 = Hadamard; H0.set_index(1);
  Gate CN = CNot;     CN.set_index(0);
  
  std::vector<Gate> layer1{H0,H1};
  std::vector<Gate> layer2{CN};
  
  std::vector<std::vector<Gate>> schedule{layer1, layer2};
  
  Circuit Bell{Input, Input, schedule};

  
  Bell.init();
  Bell.forward(handle);

  
  for (std::vector<Gate> &level  : schedule)
    for (Gate h : level )
      h.free();
  
  Input.free();

  CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));

}
