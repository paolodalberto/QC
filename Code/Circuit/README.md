# General-Purpose State-Vector Quantum-Circuit Simulation 

Take a Quantum Circuit of N q-bits. A DAG-circuit is a sequence of
Layers and it is the result of a transpilation for a defined
layout. This is a first step to map the computation to a real (or
emulated) quantum computer.

A layer is a composition of Gates applied to the circuit q-bits
(lines). The main property is that the gate computation is parallel
(not touched bits are not affected). We perform the computation by
representing the state.

q0 ----- H --- 0
q1 ------H --- X

For example, q0 and q1 are two qbits and the state is a complex vector
associated to the measurable and complete base x_i = |00>, |01>, |10>
and |11> Any vector is a composition of the Sum a_i *x_i .. a_i is a
complex number and it has the property that <A | A> =1 and a_i^* a_i
is the probability that when we measure the system the result will be |i>.

Note we specify the Kronecker product as (x) (ascii style)

The Gate computation H to bit 0 is equivalent to a
 A = [ I_2 (x) H ]*A
 
The Gate computation H to bit 1 is equivalent to a
 A = [ H (x) I_2] A

The computation is associative as long as we "accumulate" the result
and it is equivalent to the larger Heisenberg matrix for the first layer:
A = [ H (x) H ]  A

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
    
In practice, the idea is to have a circuit in the host and in the
GPU. The simulation takes an initial state and compute the final
state. This final state can be used to compute probability and thus the simulated runs.

If the Input does not change the computation is completely
deterministic unless we introduce randomness in the unitary matrix or
its computation (we could extend the gate to a injecting error gate).

** How to make it ?

First some manual work and understanding.



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