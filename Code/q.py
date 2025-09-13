from qiskit import QuantumCircuit

from qiskit.primitives import StatevectorSampler as Sampler 
#from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_aer.primitives import StatevectorSampler as Sampler

from qiskit.visualization import plot_histogram

# Create a quantum circuit with 2 qubits
bell_circuit = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit to create a superposition
bell_circuit.h(0)

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
