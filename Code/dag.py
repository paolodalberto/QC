from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.dagcircuit import DAGInNode, DAGOpNode
from qiskit.converters import circuit_to_dag
import numpy
import pdb
I2 = numpy.identity(2)
I4 = numpy.identity(4)
I8 = numpy.identity(8)


def play(transpiled_qc, minindex = 122,Bits=3):

    dag = circuit_to_dag(transpiled_qc)
    I  = numpy.identity(2**Bits) 

    T = I
    for i, layer in enumerate(dag.layers()):
    
        L = layer["graph"].op_nodes()
        Ls = sorted(L, key = lambda node: min([transpiled_qc.find_bit(q).index for q in node.qargs]))
        print(f"\nLayer {i+1} # {len(Ls)}:")
        LT = {}
        for node in Ls:
            index = min([transpiled_qc.find_bit(q).index for q in node.qargs])
            LT[index] = node
            
        G = None
        j=0
        while j < Bits:
            i = j+minindex
            j+=1
            if not i in LT:
            
                if G is None:
                    G = I2
                else:
                    G = numpy.kron(I2, G)
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
                if G is None:
                    G = matrix
                else:
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
