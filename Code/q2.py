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


## 1/4 3/4 
G = numpy.matrix(
    [
        [ numpy.cos(phi),  -numpy.sin(phi)],
        [ numpy.sin(phi),  numpy.cos(phi)]
    ]
)
print(G)

def power(A):
    return numpy.matmul(A.H,A)


# Hadamard 1/2 1/2 
alpha = 1/numpy.sqrt(2)
beta  = alpha

H = numpy.matrix(
    [
        [ 1,  1],
        [ 1, -1]
    ]
)*alpha

zero = numpy.matrix(
    [
        [ 1],
        [ 0]
    ]
)
one = numpy.matrix(
    [
        [ 0],
        [ 1]
    ]
)

H = numpy.matrix(
    [
        [ 1,  1],
        [ 1, -1]
    ]
)*alpha


g_p = UnitaryGate(G)
h_p = UnitaryGate(H)

## composition 
F = numpy.matmul(H,G)

print(F)

## if we start with |0> def = [ 1 0 ]
## G*[h_00]
##   [h_10]
## (h_00*g_00 +h_10*g_01)|0> +  (h_00*g_10 +h_10*g_11)|1>
##
## P(|0>) = (h_00*g_00 +h_10*g_01) **2
## P(|1>) = (h_00*g_10 +h_10*g_11) **2
##
##  Sum them up to check they are probability distribution 
##  h_00^2*g_00^2 +h_10^2*g_01^2 + 2h_00h_10g_00g_01 + 
##  h_00^2*g_10^2 +h_10^2*g_11^2 + 2h_00h_10g_10g_11 =
##
##  h_00^2(g_00^2+g_10^2) +  +h_10^2(g_01^2+ g_11^2) 
##  + 2h_00h_10(g_00g_01 + g_10g_11)
##
##  Because G is unitary  G^t*G = I
##  1) g_00^2+g_10^2 =  g_01^2+ g_11^2  =1
##  2)  g_00g_01 + g_10g_11 = 0 
##  
### Note there is a interference because of the terms g_00g_01 and
### g_10g_11 will be negative and thus there is interference
### decreasing the probability of either case
##  Then we have h_00^2++h_10^2 = 1  because H is unitaty
##
## if we start with |1>
## G*[h_01]
##   [h_11]
##  (h_01*g_00 +h_11*g_01)|0> +  (h_01*g_10 +h_11*g_11)|1>  
##

y = F*zero
print("estimated F*zero", y[0]**2, y[1]**2)

z = F*one
print("estimated F*one", z[0]**2, z[1]**2)

# notice we start by splitting 1/2 and 1/2 then by 1/4 and 3/4
# and we achieve 1/10 and 9/10 

# Create a quantum circuit with 1 qubits
bell_circuit = QuantumCircuit(1)

initial_state = [1, 0]
bell_circuit.initialize(initial_state, 0) 
# Apply a Hadamard gate to the first qubit to create a superposition
bell_circuit.h(0)             
bell_circuit.append(g_p, [0])

print(bell_circuit)

# Add measurements to the circuit
bell_circuit.measure_all()

# Run the circuit on a statevector simulator using the Sampler primitive
sampler = Sampler()

job = sampler.run([bell_circuit], shots=1000)
result = job.result()
counts = result[0].data.meas.get_counts()

# Print the measurement outcomes
print(counts)




