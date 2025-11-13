#!/bin/python
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time

''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
	lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 1200					# Dimension of matrix
tol = 1e-8				# Convergence tolerance
mmax = n//2				# Maximum number of iterations	

''' Create sparse, diagonally dominant matrix A with 
	diagonal containing 1,2,3,...n. The eigenvalues
    should be very close to these values. You can 
    change the sparsity. A smaller number for sparsity
    increases the diagonal dominance. Larger values
    (e.g. sparsity = 1) create a dense matrix
'''

sparsity = 0.0001
A = np.zeros((n,n))
for i in range(0,n):
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) 
A = (A.T + A)/2 


k = 8					# number of initial guess vectors 
eig = 4					# number of eignvalues to solve 
t = np.eye(n,k)			# set of k unit vectors as guess
V = np.zeros((n,n))		# array of zeros to hold guess vec
I = np.eye(n)			# identity matrix same dimen as A

# Begin block Davidson routine

start_davidson = time.time()

count = 0
for m in range(k,mmax,k):
    count +=1
    print(count)

    if m <= k:
        for j in range(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
        theta_old = 1 
    elif m > k:
        theta_old = theta[:eig]


    print("V shape ", V[:,:m].shape)
    V[:,:m],R = np.linalg.qr(V[:,:m])
    print("V shape QR Q is orthogonal", V[:,:m].shape)

    print("projection into the space")
    ## V^t * A * V
    T = np.dot(V[:,:m].T,np.dot(A,V[:,:m]))

    print("T.shape", T.shape )
    
    THETA,S = np.linalg.eig(T)
    print("Solver eigen value and vectors", THETA.shape, S.shape ) 
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:,idx]
    for j in range(0,k):
        w = np.dot((A - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
        q = w/(theta[j]-A[j,j])
        V[:,(m+j)] = q

    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break

end_davidson = time.time()

# End of block Davidson. Print results.

print("davidson = ", theta[:eig],";",
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:eig],";",
     end_numpy - start_numpy, "seconds") 
