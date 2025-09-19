# Reading Lists and Notes about them


This is in chronological order and the mispell are all mine. 

## BOOKS

### Quantum Computing (A gentle Introduction)
https://www.amazon.com/Quantum-Computing-Introduction-Engineering-Computation/dp/0262526670
this is my first read and I do not have much a memory of it.

### Mathematical Foundation of Quantum Mechanics 
This is by John von Neumann translation by Robert Beyer (not by Wheeler). I know you will make fun of me but I wanted to read Johnny's book (I have started theory of games .. but this is a different story) and a translation he read himself. Eigenstate and eigenvalue is something I can understand and the reading was a struggle. One day I may go back to try to understand it. ISBN-13 978-0691028934

### Quantum Computer Science (an introduction) 
This book is by David Mermin and I think I read this twice, the first time I Could not finish it. The second time I did by discipline alone. I think the appendices should be integrated in the text making the reading more accessable.  ISBN-13  978-0521876582

### Introduction to classical and Quantum Computing
by Thomas Wong ISBN-13 979-8985593105

This is so far the book that eased me into the QC because it relies on explanations using also classic concept. It is a fast read. There are links to youtube. The authors really wants to engage and to excite an interest for the field. It is the first book that really let you have the first experiment using Bell's inequalities. No appendices and not fluffy things, but you can see that there is substance and the decisions not to explain are clear cut and the book is completely self contained. If you want to start, Wong's book is so far the best choice. there is a full PDF available and a website for it ... 

### Quantum Computing with Python and IBM Quantum 
by Robert Loredo ISBN-13 978-1803244808

(1/2 the way through, this si a light introduction to the tools for designing, simulate, and run QC algorithms) From here I am trying to understand the building plock of the current implementation of the tools for QC. 



## ARTICLES 

### Enabling Quantum Computer Simulations on AMD GPUs: a HIP Backend for Google’s qsim
https://dl.acm.org/doi/pdf/10.1145/3624062.3624223

We are in the realm of state vector simulators, up to 30 qbit ((4+4)
complex single precision of (8+8 complex double precision) 2**30 so we
can have 8GB or 32GB), and the idea is to _hipify_ the cuda code so
that qsim just uses an on-purpose built hip device by translating the
cuda calls.  In praqctice, the basic gates are 1 and 2 bits, thus the
sparsity is quite large and as a function where the gate is located
the diagonal locations are *more* separated and strided.

The basic computation will be a matrix (transformation) by vector
(state) to obtain the next state ... and this is carried on. Gate
fusion by Kronecker and by matrix multiplication guide the
optimization.

A note about the Kronecker product (x) and parallel gates: Take H and
G two gates applied to different qbit. The step computation is
described by two matrices A = I (x) G and B = H (x) I and the overall
computation is the matrix product A * B. Then A * B = B* A = H (x) G.
The order is about the qubit and not the A and B.

Proof. A = [ [G 0] [0 G]] and B = [ [h0.I h1.I] [h2.I h3.I]]
A*B = [ [h0.G h1.G], [h2.G h3.G]] = H (x) G 
B*A = [ [h0.G h1.G], [h2.G h3.G]] 
So the computation A*B, B*A, H (x) G are all equivalent. 

In practice 8GB will fit in one GPU 32 will require some smart
division. There may be a code generation of sparse matrix vector
operation that can be encoded directly, and thus write a hip single
shot computation. But notice that the "constant matrix is literally
2x2,4x4  and the state vector is 2^n.

I think the main performance boost is by the HBM2. This is
communication bound.


### A Scalable FPGA Architecture for Quantum Computing Simulation
https://arxiv.org/pdf/2407.06415

This is an Altera implementation and it is a classic FPGA with DPS and
finite precision. The idea is brilliantly simple to describe.  We
create 1 qbit and 2qbit operators, and you create a
routing/permutation that at each step drive the 1qubit (two values in
the state) or two qbits (4 values) into the proper computational unit,
perform the complex matrix operation.  The operation is repeated to
the overall state and further parallelism can be applied here because
we can have multiple functions.  You can see this as a universal
quantum machine simulator: We need to introduce a sequence of
operations and permutations, the depth of the circuit. 

FPGA can store all operations internally and the only thing is to
traverse the state properly having a HBM should be quite appropriate.
With the proper ideas fusion of layer could actually reduce the passes
from and to the HBM.




