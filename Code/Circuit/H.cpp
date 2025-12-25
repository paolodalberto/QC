


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
 * size 2^m x 2^k ... so far brilliant stored in row major
 *
 **************/


#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hip_complex.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


// we define the computation double complex 

#define TYPE_OPERAND 4  // complex double  
#include "matrices.h"  // definition of matrices 
#include "circuit.h"   // definition of Gate and Circuit


// One day we will run on GPUs (one+)
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




extern Gate CNot;
extern const Gate Hadamard;
 

int main(int argc, char* argv[]) {
   // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = (argc>1)? std::atoi(argv[1]):0;
  int cpu       = (argc>2)? std::atoi(argv[2]):0;
  int bit       = (argc>3)? std::atoi(argv[3]):2;
  int times       = (argc>3)? std::atoi(argv[4]):1;
  int result =  set_device(mydevice);
  
  int M = 1<< bit;
  printf(" device: %d ;  State Bits %d Space %d \n", mydevice, bit, M);
  
  rocblas_handle handle;
  CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));


  Matrix Input = {M,1,M,1};
  Input.alloc(true,true);
  /*  let's try in place */
  /* Matrix Output = {M,1,M,1};
     Output.alloc(true,true);
  */
  Input.zero();
  Input.matrix[0] = ONE;
  //Input.matrix[2] = ONE/std::sqrt(2);
  Input.print(true);






  if (cpu>0) Input.writetodevice();


  // building the circuit like we belong
  std::vector<Gate> Hs;
  for (int i=0; i<bit; i++)  { 
    Gate H0 = Hadamard;
    H0.set_index(i);
    Hs.push_back(H0);
  }
  std::vector<Gate> Cs;
  for (int i=0; i<bit; i+=2)  { 
    Gate CN = CNot;
    CN.set_index(i);
    Cs.push_back(CN);
  }


  std::vector<std::vector<Gate>> schedule{Hs, Cs}; 
  
  
  Circuit Bell{Input, Input, schedule};
  Bell.init(cpu);
  Bell.print(true);
  

  Input.print(true);
  printf(" Computing \n");
  auto start_ = std::chrono::high_resolution_clock::now();

  for (int i=0; i<times;i++ ) { 
    printf("Iteration %d \n", i);
    //if (cpu>0) Input.writetodevice();
    Bell.forward_inplace(handle);
    //if (cpu>0) Input.readfromdevice();
  }


  auto end_ = std::chrono::high_resolution_clock::now();
  
  // 3. Calculate duration (e.g., in microseconds)
  auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>((end_ - start_)/times);
  
  std::cout << "Time: " << duration_.count() << " microseconds" << std::endl;


  if (cpu>0) Input.readfromdevice();
  Input.print(true);
  
  for (std::vector<Gate> &level  : schedule)
    for (Gate h : level )
      h.free();
  
  Input.free();

  CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));

}
