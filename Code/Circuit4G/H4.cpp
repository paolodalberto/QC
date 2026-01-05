#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#ifndef TYPE_OPERAND
#define TYPE_OPERAND 4
#endif

#include "matrices.h"  // definition of matrices 




/*

========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  [Model : Revision]    Temp    Power     Partitions      SCLK    MCLK     Fan     Perf  PwrCap  VRAM%  GPU%  
        Name (20 chars)       (Edge)  (Socket)  (Mem, Compute)                                                      
====================================================================================================================
0       [Radeon Pro Duo : 0x  30.0°C  11.08W    N/A, N/A        567Mhz  300Mhz   19.22%  auto  98.0W     0%   0%    
        Ellesmere [Radeon Pr                                                                                        
1       [Radeon Pro Duo : 0x  29.0°C  20.082W   N/A, N/A        849Mhz  300Mhz   100.0%  auto  98.0W     0%   0%    
        Ellesmere [Radeon Pr                                                                                        
2       [0x081e : 0xc1]       32.0°C  19.0W     N/A, N/A        808Mhz  350Mhz   19.61%  auto  250.0W    0%   0%    
        Vega 20 [Radeon VII]                                                                                        
3       [0x081e : 0xc1]       33.0°C  19.0W     N/A, N/A        808Mhz  350Mhz   19.61%  auto  250.0W    0%   0%    
        Vega 20 [Radeon VII]                                                                                        
4       [0x081e : 0xc1]       33.0°C  24.0W     N/A, N/A        808Mhz  1000Mhz  19.61%  auto  250.0W    3%   0%    
        Vega 20 [Radeon VII]                                                                                        
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================


 */



int main(int argc, char* argv[]) {
  // Method 1: Using atoi (C-style, simpler but less robust)
  printf(" CODE \n");

  int BITS  = (argc>1)? std::atoi(argv[1]):20;
  printf(" M \n");
  std::vector<Matrix>  M;
  printf(" State \n");
  STATE A{.bits = BITS, .G = M};

  printf("Init\n");
  A.init();
  
  printf("after Init\n");
  
  // This is the n+2-n bit for 4 GPU
  // 0 - 2 and 1-3
  // 0 - 1 and 2-3 is the n+1 - n bit

  // VIDEO --------------------------v
  std::vector<int> permutation{2,3,0,1};
  
  try {
    A.run_shuffle_test(permutation);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  A.free();
  return 0;
}
