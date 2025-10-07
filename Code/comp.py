import numpy
import cmath


def Il_G_Ir_state(
        Il : int , # left identity: parallelism
        G  : numpy.ndarray, # gate Computation 
        Ir : int, # right identity : either kroneker or permutation  
        state : numpy.ndarray # state
) -> numpy.ndarray : # state

    tstate = state*1
    GIR = G.shape[0]*Ir


    ##  GT = I_L (x) GQ
    ##  GQ = G (x) I_R

    ## I_L is a parallel dimension
    for i in range(Il):
        
        # this is a parallel computation where we take consecutive
        # chunks of states and we organize them as a matrix stored in
        # row major ... this is a logical change of layout
        T = tstate[i*GIR:(i+1)*GIR].reshape(G.shape[0], Ir)
       
        # then we perform the state modification as a matrix
        # multiplication. Think may of these in parallel

        R = G@T
        
        tstate[i*GIR:(i+1)*GIR] = R.flatten()
        
        ## this computation can be done in place considering if we are
        ## using Row major layout ... the former reshape is logical
        ## and the last flatten is logical.
        
    return tstate


## Heisember matrix ?
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

    return  GT@state


if __name__ == "__main__":

    CNOT = numpy.zeros((4,4),dtype=complex)
    CNOT[0,0] = complex(1,0) 
    CNOT[1,1] = complex(1,0) 
    CNOT[2,3] = complex(1,0) 
    CNOT[3,2] = complex(1,0) 
    angle_rad = numpy.pi / 16
    
    invsqrt2 = numpy.sqrt(2)
    
    state = numpy.zeros(2*4*2,dtype=complex)
    for i in range(2*4*2):
        state[i] =  invsqrt2*complex(1,-1)* cmath.rect(1, angle_rad*i)
    
    y = Kronecker_Il_G_Ir_state(2,CNOT, 2, state)
    print(y.shape,state.shape)


    y1 = Il_G_Ir_state(2,CNOT, 2, state)
    print(y1.shape)
    #import pdb; pdb.set_trace()
    print(y-y1)
    #import pdb; pdb.set_trace()
    
