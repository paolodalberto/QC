from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.dagcircuit import DAGInNode, DAGOpNode

from qiskit.converters import circuit_to_dag
from qiskit.primitives import StatevectorSampler as Sampler 
#from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_aer.primitives import StatevectorSampler as Sampler

from qiskit.visualization import plot_histogram

from qiskit.circuit.library import UnitaryGate

import numpy
phi = 2*numpy.pi/3


# Gemini prompt 
# an example of unitary gate that split the probability 1/4 and 3/4

# An example of a unitary gate that splits a probability into a 1/4 and a 3/4 chance is a rotation gate.
# A rotation about the y-axis, \R_{y}(\theta ), can transform an initial state |0> into a superposition of |0> and |1>.
# For an initial state of |0>, the resulting state is given by the first column of the rotation matrix:
#
# R_{y}(\theta)|0> =\begin{matrix}\cos (\theta /2)\ \sin (\theta /2)\end{matrix}.
#
# To achieve probabilities of 1/4 and 3/4, the squared magnitudes of
# the amplitudes must be \(|\cos (\theta /2)|^{2}=1/4\) and \(|\sin (\theta /2)|^{2}=3/4\).

# This gives a rotation angle of \(\theta \) where \(\cos (\theta /2)=\pm 1/2\) and \(\sin (\theta /2)=\pm \sqrt{3}/2\). A common
# solution is \(\theta /2=\pi /3\), which means \(\theta =2\pi /3\).
#
# The unitary matrix for this gate 

matrix = numpy.matrix(
    [
        [ numpy.cos(phi),  -numpy.sin(phi)],
        [ numpy.sin(phi),  numpy.cos(phi)]
    ]
)
g_p = UnitaryGate(matrix)
 


# Create a quantum circuit with 2 qubits
bell_circuit = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit to create a superposition
bell_circuit.h(0)
bell_circuit.append(g_p, [1])

# Apply a CNOT gate to entangle the qubits
bell_circuit.cx(0, 1)

# Add measurements to the circuit
bell_circuit.measure_all()
print(bell_circuit)

simulator = AerSimulator()
backend = FakeSherbrooke()

transpiled_qc = transpile(bell_circuit,  backend=backend, optimization_level=3)
print("\nTranspiled Circuit:")
print(transpiled_qc.draw())


# Convert the transpiled circuit to a DAGCircuit
dag = circuit_to_dag(transpiled_qc)

print("Final transpiled DAG properties:")
print(f"Number of operations: {len(dag.op_nodes())}")
print(f"Final gate counts: {dag.count_ops()}")
print(f"Number of qubits: {dag.num_qubits()}")

print("--- Visiting DAG nodes in topological order ---")
for node in dag.topological_op_nodes():
    operation = node.op
    qubits = node.qargs
    print(f"Operation: {operation.name}, Qubits: {qubits}")

import pdb; pdb.set_trace()

print("\n--- Traversing DAG using ancestors and successors ---")
input_nodes = dag.input_map
first_qubit = dag.qubits[0]
current_node  = input_nodes[first_qubit]

print(f"Starting traversal from input node: {current_node.wire}")

# Traverse successors until an output node is reached
while isinstance(current_node, (DAGInNode, DAGOpNode)):
    successors = dag.successors(current_node)
    
    # Find the next node on the same wire
    next_node = None
    for successor in successors:
        print(f"  -> Reached successor {successor.wire}", type(successor))
        if isinstance(successor, DAGOpNode) and first_qubit in successor.qargs:
            next_node = successor
            break
        elif isinstance(successor, DAGOpNode) and any(
            arg == current_node.wire for arg in successor.qargs
        ):
            next_node = successor
            break
            
    if not next_node:
        # Move to the final output node if no more operations are found
        output_node = dag.output_map[first_qubit]
        print(f"  -> Reached output node for qubit {output_node.wire}")
        break

    if isinstance(next_node, DAGOpNode):
        op_name = next_node.op.name
        qubits = [q.index for q in next_node.qargs]
        print(f"  -> Operation: {op_name}, Qubits: {qubits}")
    
    current_node = next_node

pdb.set_trace()

print("Final layout:", final_layout)

# Run the circuit on a statevector simulator using the Sampler primitive
sampler = Sampler()

job = sampler.run([bell_circuit], shots=1000)
result = job.result()
counts = result[0].data.meas.get_counts()

# Print the measurement outcomes
print(counts)

simulator = AerSimulator()

job = simulator.run([transpiled_qc], shots=1000)
result = job.result()
counts = result.get_counts(transpiled_qc)

print(counts)
input_nodes = dag.input_map
