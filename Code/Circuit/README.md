# General-Purpose State-Vector Quantum-Circuit Simulation 

Take a Quantum Circuit of N q-bits. A DAG-circuit is a sequence of
Layers and it is the result of a transpilation for a defined
layout. This is a first step to map the computation to a real (or
emulated) quantum computer.

A layer is a composition of Gates applied to the circuit q-bits
(lines). The main property is that the gate computation is parallel
(not touched bits are not affected). We perform the computation by
representing the state.
```
q0 ----- H --- 0
q1 ------H --- X
```
For example, q0 and q1 are two qbits and the state is a complex vector
associated to the measurable and complete base x_i = |00>, |01>, |10>
and |11> Any vector is a composition of the Sum a_i *x_i .. a_i is a
complex number and it has the property that <A | A> =1 and a_i^* a_i
is the probability that when we measure the system the result will be |i>.

Note we specify the Kronecker product as (x) (ascii style)

The Gate computation H to bit 0 is equivalent to a
```
A = [ I_2 (x) H ]*A
```
 
The Gate computation H to bit 1 is equivalent to a
```
A = [ H (x) I_2] A
```

The computation is associative as long as we "accumulate" the result
and it is equivalent to the larger Heisenberg matrix for the first layer:
```
A = [ H (x) H ]  A
```

A 1 qbit Gate is a 2x2 Unitary matrix gate (Hadamard)

A 2 qbit Gate is a 4x4 Unitary matrix gate (CNOT, H (x) H ) 

A 3 qbit Gate is a 8x8 Unitary matrix gate (CCNOT)

The number of layers determines the depth of the circuit. Each Layer
is a collection of Gates. We can represent a Gate as *Unitary* complex
matrix. This prospective in general is not optimal: CNOT is a 4x4
matrix but it is composed of 4 1s (diagonals) and 8 0s ... so it is
sparse and the computation is literally a controlled swap. As a
complex matrix the CNOT * A it is a 4x4 (size) * 4 (double operations).

In this directory we show how to implement the general computation as
a flexible matrix by vector operations. Thus every Gate is composed of
a Matrix **U** and it applies a specific matrix vector update to the
state as a function of the "bits" applied to. We assume this is a
transpilation and the bits are actually contiguous (thanks to swaps
and optimizations). 

As Data Structures:

* Matrix is the basic structure (matrices.h) and we represent vector
  as matrices with columns =1.
  * A matrix has a pointer for the CPU and one for the GPU
  * We define a algebra (*,+) using gemm and geam operation for CPU and GPU
* We have a Gate,
  * Unitary Matrix (one copy)
  * Input and output Matrix
  * bit where we apply the gate
  * Step function defining O = F(U,I), which is usually in place I=O
    * this is CPU, GPU two implementations
  * a little of book keeping.
* We have a Circuit 
  * A vector of Layers
    * A layer is a vector of Gates

Notice, This is a general Gate description and implementation. We can
specialize any Gate and its gate computation by instead of a function
for the computation a pointer to a function. At this time, we stick to
a general computation especially because we are just trying to
understand the basic mathematics of the state update and computation. 
    
In practice, the idea is to have a circuit in the host and in the
GPU. The simulation takes an initial state and computes the final
state. This final state can be used to compute probability and thus the simulated runs.

If the Input does not change the computation is completely
deterministic; unless, we introduce randomness in the unitary matrix
or its computation (we could extend the gate to a injecting error
gate).

** How to make it ?

There are two source files: H.cpp  circuit.cpp 
and two include files: matrices.h and circuit.h
We use two BLAS libraries : openblas (CPU) and rocblas (GPU) 



Either introduce the definition in the Make file or change it in the
sources: single precision and double precision complex have to be used
(we tested only double complex).

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

Then make it 

```bash
f-docker /scratch/QC/Code/Circuit > make 
------------------------------------------------------
Compiling H.cpp to object file H.o
/opt/rocm/bin/hipcc -O1 --std=c++17 -I./  -I./  -c H.cpp -o H.o
------------------------------------------------------
------------------------------------------------------
Compiling circuit.cpp to object file circuit.o
/opt/rocm/bin/hipcc -O1 --std=c++17 -I./  -I./  -c circuit.cpp -o circuit.o
------------------------------------------------------
------------------------------------------------------
Linking object files into the final executable: H
/opt/rocm/bin/hipcc -O1 --std=c++17 -I./ H.o circuit.o -lrocblas -lopenblas -lpthread -lm -o H
------------------------------------------------------
```


then to execute you may want to be aware of a few parameters
```c++
int main(int argc, char* argv[]) {
   // Method 1: Using atoi (C-style, simpler but less robust)
  int mydevice  = (argc>1)? std::atoi(argv[1]):0; // GPU device #
  int cpu       = (argc>2)? std::atoi(argv[2]):0; // cpu=0, GPUGEMM cpu=1, Batched cpu=2, test cpu=3
  int bit       = (argc>3)? std::atoi(argv[3]):2; // number of bits
  int times     = (argc>4)? std::atoi(argv[4]):1; // number of time we repeat the computation
  int test      = (argc>5)? std::atoi(argv[5]):0; // test=0 to use Hadamard and CNOT test=1 for complex computation only (no compiler can optimize these matrices).
 

```



```bash
tf-docker /scratch/QC/Code/Circuit > ./H 0 2 25 1 0            
Device 0: AMD Instinct MI100
Device 1: AMD Instinct MI100
Device 2: AMD Instinct MI100
Successfully set device to 0
 device: 0 ;  State Bits 25 Space 33554432 
Column Major M,N = 33554432,1
(1,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
(0,0) 
 Computing 
Iteration 0 
Time: 0.040508 Ops 13421772800 TFlops: 0.331337
Time: 0.000234661 Ops 13958643712 TFlops: 59.4843
Average Time: 0.258397 Ops 27380416512 TFlops: 0.105962
BEGIN Circuit 2 
Level 0 < 25 
>>>>>> Gate hadamard
Calls 1 Bit 0  Batch 16777216 
OPS  536870912.000000 TFLOPS  0.013441 Time 0.039942 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 1  Batch 8388608 
OPS  536870912.000000 TFLOPS  7.615946 Time 0.000070 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 2  Batch 4194304 
OPS  536870912.000000 TFLOPS  34.261066 Time 0.000016 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 3  Batch 2097152 
OPS  536870912.000000 TFLOPS  29.720489 Time 0.000018 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 4  Batch 1048576 
OPS  536870912.000000 TFLOPS  30.343690 Time 0.000018 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 21  Batch 8 
OPS  536870912.000000 TFLOPS  36.304498 Time 0.000015 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 22  Batch 4 
OPS  536870912.000000 TFLOPS  28.158550 Time 0.000019 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 23  Batch 2 
OPS  536870912.000000 TFLOPS  10.216577 Time 0.000053 
Column Major M,N = 2,2
<<<<<< Gate hadamard
>>>>>> Gate hadamard
Calls 1 Bit 24  Batch 1 
OPS  536870912.000000 TFLOPS  4.920681 Time 0.000109 
Column Major M,N = 2,2
<<<<<< Gate hadamard
Level 1 < 13 
>>>>>> Gate CNot
Calls 1 Bit 0  Batch 8388608 
OPS  1073741824.000000 TFLOPS  68.570268 Time 0.000016 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 2  Batch 2097152 
OPS  1073741824.000000 TFLOPS  84.122675 Time 0.000013 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 4  Batch 524288 
OPS  1073741824.000000 TFLOPS  85.810103 Time 0.000013 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 6  Batch 131072 
OPS  1073741824.000000 TFLOPS  100.256006 Time 0.000011 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 8  Batch 32768 
OPS  1073741824.000000 TFLOPS  92.388730 Time 0.000012 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 18  Batch 32 
OPS  1073741824.000000 TFLOPS  71.592334 Time 0.000015 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 20  Batch 8 
OPS  1073741824.000000 TFLOPS  73.756136 Time 0.000015 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 22  Batch 2 
OPS  1073741824.000000 TFLOPS  53.992147 Time 0.000020 
Column Major M,N = 4,4
<<<<<< Gate CNot
>>>>>> Gate CNot
Calls 1 Bit 24  Batch 1 
OPS  1073741824.000000 TFLOPS  20.854620 Time 0.000051 
Column Major M,N = 4,4
<<<<<< Gate CNot
END Circuit 
Column Major M,N = 33554432,1
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
(0.000172633,0) 
OPS L1 536870912  L2 0 
```

This is an example where we run the batched computation. We annotated
each gate with performance information but they may give you only an
idea because the function calls are ASYNC for batched and not at all
for GEMM.

The parameters ALPHA and BETA for the GEMM computation are from the
host and we can improve these synchronizations.

** Considerations **

The CPU implementation is based on openBLAS. There are other libraries
that will take advantage of vector computations and this should make
the CPU faster.

The GPU based on rocblas GEMM is worse that the CPU implementation.

The GPU based on batched GEMM is ASYNC but not completely (Alpha and
beta). As of this writing I could not make it work to have a copy
resident in the GPU and have a correct computation (I will need more
help).

In practice, batched GPU implementation can be 100x faster that the
CPU for the whole graph ... Please take a look at the graph
construction here and in the code

```c++
  // building the circuit like we belong
  std::vector<Gate> Hs;
  for (int i=0; i<bit; i++)  { 
    Gate H0 = (test==0)?Hadamard:TTwo;
    H0.set_index(i);
    Hs.push_back(H0);
  }
  std::vector<Gate> Cs;
  for (int i=0; i<bit; i+=2)  { 
    Gate CN = (test==0)?CNot:TFour;
    CN.set_index(i);
    Cs.push_back(CN);
  }

  std::vector<std::vector<Gate>> schedule{Hs, Cs}; 

```

