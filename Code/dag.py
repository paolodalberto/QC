from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.dagcircuit import DAGInNode, DAGOpNode
from qiskit.converters import circuit_to_dag
import numpy
import pdb


### the basic identities we can use for the coputation of the
### kronecker matrix

I2 = numpy.identity(2)
I4 = numpy.identity(4)
I8 = numpy.identity(8)

                   

##  Take a transpiled graph ... this will be kind of embedded into a
##  larger set of qubits. This is to play with the DAG and create the
##  transfer matrix .. so that we can multiply this by the input state
##  and get the probability of the output

def Transfer_matrix_play(
        transpiled_qc,  
        minindex = 122,
        Bits=3):

    # transform the transpiled into a DAG
    dag = circuit_to_dag(transpiled_qc)

    #  Starting matrix 
    I  = numpy.identity(2**Bits) 

    T = I

    ## this si the depth of the circuit and each operation in this
    ## layer is actually "parallel" for a lack of a better word.

    for i, layer in enumerate(dag.layers()):
        
        # to construct the correct matrix we need to keep the right order 
        L = layer["graph"].op_nodes()
        Ls = sorted(L, key = lambda node: min([transpiled_qc.find_bit(q).index for q in node.qargs]))
        print(f"\nLayer {i+1} # {len(Ls)}:")
        LT = {}
        for node in Ls:
            index = min([transpiled_qc.find_bit(q).index for q in node.qargs])
            LT[index] = node
            
        G = 1 # Starting llittle transfer matrix that is a constant
        j=0
        while j < Bits:
            i = j+minindex
            j+=1
            if not i in LT:
                G = numpy.kron(I2, G)
                else:          G = 
                print(G.shape)
                print("skip")
                continue
            
            node = LT[i]
            print(f"  - Operation: {node.op.name}, Qubits: {node.qargs}")
            instruction = node.op
            if hasattr(instruction, 'to_matrix'):
                indexes =  [transpiled_qc.find_bit(q).index for q in node.qargs]
                matrix = instruction.to_matrix()
                j+=len(indexes)-1

                print("indexes", indexes)
                print("matrix", numpy.round(matrix.flatten(), 3))
                print("(x)", numpy.round(G, 3))
                    
                
                G = numpy.kron(matrix, G)
                print("(x)", numpy.round(G, 3))
                
            else:
                break
        if not G is None:
            print("update T", G.shape)
            #numpy.round(G, 3))
            T = T@G


    return T
