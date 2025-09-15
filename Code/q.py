from qiskit import QuantumCircuit

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

matrix = [
    [ numpy.cos(phi),  -numpy.sin(phi)],
    [ numpy.sin(phi),  numpy.cos(phi)]
]
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

# Run the circuit on a statevector simulator using the Sampler primitive
sampler = Sampler()

job = sampler.run([bell_circuit], shots=1000)
result = job.result()
counts = result[0].data.meas.get_counts()

# Print the measurement outcomes
print(counts)
