# Davidson Algorithm

This is an example of translation of Davidson algorithm using ROCM and
all in GPU, even the comparison test.  Also this is an exercize for
different types: float,double, float complex, and double complex.

Either introduce the definition in the Make file or change it in the davidson.h file

```c++
/**
 *  TYPE_OPERAND 0 HALF precision ... not working
 *  TYPE_OPERAND 1 float
 *  TYPE_OPERAND 2 double 
 *  TYPE_OPERAND 3 float complex 
 *  TYPE_OPERAND 0 double complex
 */ 

#define TYPE_OPERAND 4

```

Then make it and run it. 



```bash

paolod@xsjfislx31:/scratch/Quant/QC/Code/DavidsonR/C$ make all
/opt/rocm/bin/hipcc    -c -o sortarg.o sortarg.cpp
/opt/rocm/bin/hipcc    -c -o preconditioning.o preconditioning.cpps

/opt/rocm/bin/hipcc    -c -o davidson.o davidson.cpp
------------------------------------------------------
Linking object files into the final executable: davidson
/opt/rocm/bin/hipcc  sortarg.o preconditioning.o davidson.o -lrocblas -lrocsolver -o davidson
------------------------------------------------------
```
then to execute
```bash
## davidson Device-number Size Iterations
paolod@xsjfislx31:/scratch/Quant/QC/Code/DavidsonR/C$   ./davidson 0 45000 2 10
[sudo] password for paolod: 
Integer from atoi: 0
Device 0: AMD Instinct MI100
Device 1: AMD Instinct MI100
Device 2: AMD Instinct MI100
Successfully set device to 0
 device: 0 M: 45000 n_eng: 2 it: 10 
 davidson  2 10 
Davidson converged after 5 iteration
 0 8.693980e-01 
 1 2.000476e+00 
davidson: 4.54741 seconds.
```

In the code there is a validation if you like.