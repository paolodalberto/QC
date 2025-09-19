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
bell_circuit = QuantumCircuit(3)

# Apply a Hadamard gate to the first qubit to create a superposition
bell_circuit.h(0)
bell_circuit.append(g_p, [1])

# Apply a CNOT gate to entangle the qubits
bell_circuit.cx(0, 1)
bell_circuit.cx(0, 2)

# Add measurements to the circuit
bell_circuit.measure_all()
print("Circuit")
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
    print(f"  - Operation: {node.op.name}, Qubits: {node.qargs}")
    
import pdb; pdb.set_trace()

print("--- Traversing the DAG layer by layer ---")
for i, layer in enumerate(dag.layers()):
    print(f"\nLayer {i+1}:")
    for node in layer["graph"].op_nodes():
        print(f"  - Operation: {node.op.name}, Qubits: {node.qargs}")

pdb.set_trace()

# Run the circuit on a statevector simulator using the Sampler primitive
sampler = Sampler()

job = sampler.run([bell_circuit], shots=1000)
result = job.result()
counts = result[0].data.meas.get_counts()

# Print the measurement outcomes
print(sorted(counts.items(),key = lambda x : x[0]))

simulator = AerSimulator()

job = simulator.run([transpiled_qc], shots=1000)
result = job.result()
counts = result.get_counts(transpiled_qc)

print(sorted(counts.items(),key = lambda x : x[0]))
input_nodes = dag.input_map
