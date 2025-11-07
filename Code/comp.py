######################################################################################################
###
### STATE WISE COMPUTATION (this will be able to handle very limited #bits)
### 
######################################################################################################


import numpy
import cmath
import lib


##############################################################
###
### STATE as SINGLE VECTOR
###
##############################################################


########
### 
### We have a gate G as a matrix (1bit) 2x2, (2bit) 4x4, (3bit) 8x8
### connecting consecutive gl-consecutive bits.
###
### We have T = Il + gl + Ir total bits and this is the pseudo circuit
###
###  ------ Il ---- 
###  ------ G  ----
###  ------ Ir ---- 
###
### If we take T bits, the state is a column vector S = 2^T wide and
### the computation is the matrix by vector A * S, where
###
### A is I_L (x) (G (x) I_R) and (x) is the Kronecker product
###
### I think this is called the Heisenberg matrix. Clearly A is a
### 2^Tx2^T matrix and 2^T is already large. This is the reference in
### the sense that this is description of the computation and Quantum
### algorithm (or equivalent to the circuit/algorithm). 
########

def Kronecker_Il_G_Ir_state(
        Il : int , # left identity: parallelism
        G  : numpy.ndarray, # gate Computation 
        Ir : int, # right identity : either kroneker or permutation  
        state : numpy.ndarray # state
) -> numpy.ndarray : # state

    ## 
    I_L = numpy.identity(Il)
    I_R = numpy.identity(Ir)

    
    ##  GT = I_L (x) (G (x) I_R)
    GT   = numpy.kron(I_L,  numpy.kron(G,I_R))

    ## The creation of GT may seem unnecessary but it is a true method
    
    return  GT@state

#######
### Instead of creating A= I_L (x) (G (x) I_R), we just use G
### only. The basic computation is a ZGEMM G * X where we read and
### logically organized the input state into a sequence gl x matrices. 
###
### The space complexity of this operation is G (2^gl) and we can
### build the update state as described below.
###
#######

def Il_G_Ir_state(
        Il : int , # left identity: parallelism
        G  : numpy.ndarray, # gate Computation 
        Ir : int, # right identity : either kroneker or permutation  
        state : numpy.ndarray # state
) -> numpy.ndarray : # state

    tstate = state*1
    GIR = G.shape[0]*Ir

    ##  print(tstate.shape)
    ##  GT = I_L (x) GQ
    ##  GQ = G (x) I_R

    ## I_L is a parallel dimension, this is slow because of python
    for i in range(Il):
        
        # This is a parallel computation where we take consecutive
        # chunks of states and we organize them as a matrix stored in
        # row major ... this is a logical change of layout.
        

        # the state modification is a matrix multiplication. 
        # However, G has 2,4,8 rows and Ir is a power of 2.
        #
        # For G = 2 we must have an associativity >=2 
        # For G = 4 we must have an associativity >=4 
        # For G = 8 we must have an associativity >=8
        #
        # because Ir is a large power of two and a cache line may not
        # be used. otherwise the is a lot of spatial locality and we
        # can use vector instructions instead of ZGEMM kernel 


        T = tstate[i*GIR:(i+1)*GIR].reshape(G.shape[0], Ir)

        R = G@T 
        
        tstate[i*GIR:(i+1)*GIR] = R.flatten()
        
        ## Note: this computation can be done IN PLACE considering if
        ## we are using Row major layout ... the former reshape is
        ## logical and the last flatten is logical.
        
    return tstate



## the partition of the computation is by columns the state will be
## collected as a single vector. but w can play with the idea that we
## build the state on the fly by P

def G_state_P(
        G  : numpy.ndarray, # gate Computation 
        state : numpy.ndarray, # state
        P : int = 2**4
) -> numpy.ndarray : # state

    Ir = state.shape[0]//G.shape[0]
    K = Ir//P
    T = state.reshape(G.shape[0], Ir)
    for i in range(P):
        T[:,i*K:(i+1)*K] = G@T[:,i*K:(i+1)*K] 
        
    state = T.flatten()
    return state




#####
##
## Missing implementation is the case the Gate does not hit
## consecutive bits G2 for example hit bit 1 and 3 covering a space of
## 3 bits or 8 bits.
##
## In general, we introduce swaps (contiguous gates). These are gates
## that have to happen in the previous step and they may be combined
## into a single gate.
##
##  -- -  -- G --   
##  -- X  -- G -- X
##  -- X  -- - -- X  
##      
##   swap       swap 
##
## In this context, we consider the Circuit coming from a transpile
## and thus layer and swap are already in but we may consider the
## application of "matrix matrix multiplication per layer so to
## combine layer operations ... I do not know how it works.
##
## I can see something like
##  -- M - G  -- becoming -- M*G --
##  -- M - G  -- 
##
## So may be I do not need to worry too much at this point this
## belongs to the DAG optimizations ?
####


##############################################################
###
### STATE as MULTIPLE VECTORS (Distributed)
###
##############################################################


###
## The state above is a single vector. The Heisenberg matrix is a
## single vector. This is not possible in general because the state is
## 2^n we need to split the space in multiple parts just to store it
## or to compute the state efficiently.
##
##
## This simulate a distributed state as a class. We will try to reuse
## the computations above as much as possible ... they assume the
## state as a single vector (single location)
###

class State:

    def __init__(self, state : list):
        self.state = [] 
        space  = 0
        
        for i in state:
            self.state.append(i) 
            space += i.shape[0]


        self.bits_p  = int(numpy.log2(self.state[0].shape[0]))
        self.bits_top = int(numpy.log2(len(self.state)))
        self.bits     =    self.bits_p +   self.bits_top    


    ## The computation is practically
    ## X = x1 |  x2
    ## G*X where X is K x N/K 
    ## G * [ A is K x N/K/2 | B] 
    ## 
        
    def case_2(x1, x2, G):

        ## temporary space 
        
        N = x1.shape[0]
        K = G.shape[0]

        X1 = x1.reshape(K, N/K)
        X2 = x2.reshape(K, N/K)
        
        T1 = (x1*1) # copy  
        T2 = (x2*1) # or local temporary buffer

        M = K/2

        ## permute 
        T1[M: ,:] = X2[0:M,:]
        T2[0:M,:] = X1[M: ,:]


        T1 =  G@T1 # G >= 2 
        T2 =  G@T2 

        # permute back 
        X1[0:M,:]  = T1[0:M,:]
        X1[M: ,:]  = T2[0:M,:]
        X2[0:M,:]  = T1[M: ,:]
        X2[M: ,:]  = T2[M: ,:]



    def case_4(x1, x2,
               x3, x4,
               G):

        N = x1.shape[0]
        K = G.shape[0]

        if K==2:
            ## two parallel
            case_2(x1,x2,G) 
            case_2(x3,x4,G)
            
        else:
        
            X0 = x1.reshape(K, N/K) # This is a logical reshape
            X1 = x2.reshape(K, N/K)
            X2 = x3.reshape(K, N/K)
            X3 = x4.reshape(K, N/K)
        
            T0 = (X0*1) # copy  
            T1 = (X1*1) # or local temporary buffer
            T2 = (X2*1) # copy  
            T3 = (X3*1) # or local temporary buffer
        
            T0[1,:]   = X1[0,:]
            T0[2,:]   = X2[0,:]
            T0[3,:]   = X3[0,:]

            T1[0,:]   = X0[1,:]
            T1[2,:]   = X2[1,:]
            T1[3,:]   = X3[1,:]

            T2[0,:]   = X0[2,:]
            T2[1,:]   = X1[2,:]
            T2[3,:]   = X3[2,:]

            T3[0,:]   = X0[3,:]
            T3[1,:]   = X1[3,:]
            T3[2,:]   = X2[3,:]


            T0 =  G@T0  ## PARALLEL
            T1 =  G@T1  ## PARALLEL
            T2 =  G@T2 
            T3 =  G@T3 
            
            X0[0,:]  = T0[0,:]
            X1[0,:]  = T0[1,:]
            X2[0,:]  = T0[2,:]
            X3[0,:]  = T0[3,:]

            X0[1,:]  = T1[0,:]
            X1[1,:]  = T1[1,:]
            X2[1,:]  = T1[2,:]
            X3[1,:]  = T1[3,:]

            X0[2,:]  = T2[0,:]
            X1[2,:]  = T2[1,:]
            X2[2,:]  = T2[2,:]
            X3[2,:]  = T2[3,:]

            X0[3,:]  = T3[0,:]
            X1[3,:]  = T3[1,:]
            X2[3,:]  = T3[2,:]
            X3[3,:]  = T3[3,:]

    def case_8(x1, x2,
               x3, x4,
               x5, x6,
               x7, x8,
               G):

        N = x1.shape[0]
        K = G.shape[0]

        if K==4:
            ## two parallel
            case_4(x1,x2,x3,x4,G) 
            case_4(x5,x6,x7,x8,G)
            
        else:
        
            X0 = x1.reshape(K, N/K) # This is a logical reshape
            X1 = x2.reshape(K, N/K)
            X2 = x3.reshape(K, N/K)
            X3 = x4.reshape(K, N/K)
        
            T0 = (X0*1) # copy  
            T1 = (X1*1) # or local temporary buffer
            T2 = (X2*1) # copy  
            T3 = (X3*1) # or local temporary buffer
        
            T0[1,:]   = X1[0,:]
            T0[2,:]   = X2[0,:]
            T0[3,:]   = X3[0,:]

            T1[0,:]   = X0[1,:]
            T1[2,:]   = X2[1,:]
            T1[3,:]   = X3[1,:]

            T2[0,:]   = X0[2,:]
            T2[1,:]   = X1[2,:]
            T2[3,:]   = X3[2,:]

            T3[0,:]   = X0[3,:]
            T3[1,:]   = X1[3,:]
            T3[2,:]   = X2[3,:]


            T0 =  G@T0  ## PARALLEL
            T1 =  G@T1  ## PARALLEL
            T2 =  G@T2 
            T3 =  G@T3 
            
            X0[0,:]  = T0[0,:]
            X1[0,:]  = T0[1,:]
            X2[0,:]  = T0[2,:]
            X3[0,:]  = T0[3,:]

            X0[1,:]  = T1[0,:]
            X1[1,:]  = T1[1,:]
            X2[1,:]  = T1[2,:]
            X3[1,:]  = T1[3,:]

            X0[2,:]  = T2[0,:]
            X1[2,:]  = T2[1,:]
            X2[2,:]  = T2[2,:]
            X3[2,:]  = T2[3,:]

            X0[3,:]  = T3[0,:]
            X1[3,:]  = T3[1,:]
            X2[3,:]  = T3[2,:]
            X3[3,:]  = T3[3,:]

        
    def  Il_G_Ir_state(self,
            Il : int ,          # left bits 
            G  : numpy.ndarray, # gate Computation 
            Ir : int):          # right bits
        
        #import pdb; pdb.set_trace()
        bitgate = int(numpy.log2(G.shape[0]))
        if Il+bitgate+Ir > self.bits:
            # this operation has more bits than the physical one
            return 0
        
        ## every thing fit into a single part 
        if Il>= self.bits_top or bitgate+Ir <= self.bits_p:
            print("distribute the computation")
            for s in self.state:
                ## each state is distibuted into different spaces you
                ## can think of GPUs but I simulate them by different
                ## arrays
                
                s = Il_G_Ir_state(2**(self.bits_p-bitgate-Ir), G, 2**Ir,s)
            return 1
         
        ## it does not fit into a single part
        ## "permutation ?


        K =   bitgate+Ir - self.bits_p

        ## gather
        S = numpy.zeros(2**self.bits, dtype = self.state[0].dtype)
        Q =self.state[0].shape[0]
        L = len(self.state)
        print("gather", L )
        for s in range(L):
            S[s*Q:(s+1)*Q] =self.state[s]

        print("Computation")
        G_state_P(G, S)
        print("scatter", L )
        for s in range(L):
             self.state[s] = S[s*Q:(s+1)*Q]
        

        del S
        return 1

    
def two_state(
        E : int = 20,
        CNOT =  lib.cnot(),
        K : int = 1):
    
    gbits = int(numpy.log2(CNOT.shape[0]))

    I =  2**(E+gbits)
    state = [] 
    for i in range(2**K) :
        print("state", i)
        state.append(numpy.ones(I,dtype=complex))

    ET = E+gbits +K
    print("bits simulation", ET)
    S = State(state)
    head = 0
    start_time = time.perf_counter()
    for i in range(5):
        S.Il_G_Ir_state(head,CNOT,ET-head-gbits) 
    end_time = time.perf_counter()
    # Calculate the elapsed time
    elapsed_time = (end_time - start_time)/5
    print(f"Average time: {elapsed_time:.6f} seconds")







if __name__ == "__main__":


##############################################################
###
### TESTS
###
##############################################################
    SINGLE   = False 
    MUTLIPLE = True #False 

    import time
    if MUTLIPLE:
        two_state(27,lib.cnot(),1)
    

    if SINGLE:
        E = 20
        I =  2**E
        CNOT =lib.cnot()
        
        state = numpy.ones(I*4,dtype=complex)
    

    
        #y = Kronecker_Il_G_Ir_state(I_l,CNOT, I_r, state)
        #print(y.shape,state.shape)
        
        
        for i in range(1,E,1):
            print(i,CNOT.shape[0], E-i)
            # Record the start time
            start_time = time.perf_counter()
            y1 = Il_G_Ir_state(2**i,CNOT, 2**(E-i), state)
            #print(y1.shape, y1[1000])
            end_time = time.perf_counter()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.6f} seconds")


    
