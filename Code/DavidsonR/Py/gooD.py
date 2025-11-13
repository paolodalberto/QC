import numpy as np

def davidson_algorithm(A, num_eigs, max_iterations=100, tolerance=1e-8):
    """
    Basic implementation of the Davidson algorithm for a symmetric, 
    diagonally dominant matrix A.
    
    Args:
        A (np.ndarray or function): The matrix (or function performing matrix-vector product).
        num_eigs (int): Number of lowest eigenvalues to compute.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
    """
    
    n = A.shape[0]
    # Check if A is a numpy array, if not, assume it's a function (matrix-free)
    if isinstance(A, np.ndarray):
        mat_vec_prod = lambda v: np.dot(A, v)
        # Preconditioner (diagonal inverse)
        preconditioner = lambda r, eig_val: r / (np.diag(A) - eig_val + 1e-12)
    else:
        # For matrix-free implementations, you need a custom preconditioner
        raise NotImplementedError("Matrix-free mode requires a specific preconditioner function.")

    # Initial guess vectors (e.g., identity matrix columns corresponding to lowest diagonal elements)
    # Sort diagonal indices to pick the ones corresponding to the lowest expected eigenvalues
    diag_indices = np.argsort(np.diag(A))
    V = np.zeros((n, num_eigs))
    for i in range(num_eigs):
        V[diag_indices[i], i] = 1.0
    
    # Orthonormalize initial subspace
    V, _ = np.linalg.qr(V)

    for iteration in range(max_iterations):

        T = V.T @ A @ V # ZGEMM ZGEMM
        # Project A onto the subspace: T = V^T * W
        
        print("Solver to T", T.shape)
        # Solve the small eigenvalue problem for T
        eig_vals_T, eig_vecs_T = np.linalg.eigh(T) # rocsolver_dsyev_ 
        
        # Sort eigenvalues and corresponding eigenvectors
        idx = np.argsort(eig_vals_T)
        eig_vals_T = eig_vals_T[idx]
        eig_vecs_T = eig_vecs_T[:, idx]
        
        # Select the lowest 'num_eigs' solutions (Ritz values/vectors)
        current_eig_vals = eig_vals_T[:num_eigs]

        # current_eig_vecs = np.dot(V, eig_vecs_T[:, :num_eigs]) # Ritz vectors in full space
        current_eig_vecs = V @ eig_vecs_T[:, :num_eigs] # Ritz vectors in full space
        
        
        # Calculate residuals R = A*u - lambda*u = W*c - lambda*V*c
        # Easier formula: R = W_k - lambda_k * V_k (where V_k is current ritz vector in full space)

        #residuals = np.zeros((n, num_eigs))
        converged_count = 0
        #  ZGEMM  and ZGEMA 
        residuals = A @ current_eig_vecs -   current_eig_vals * current_eig_vecs
        
        
        for i in range(num_eigs):
            if np.linalg.norm(residuals[:, i]) < tolerance:
                converged_count += 1
        
        if converged_count == num_eigs:
            print(f"Davidson converged after {iteration + 1} iterations.")
            return current_eig_vals, current_eig_vecs

        # Compute correction vectors (preconditioning step)
        corrections = np.zeros((n, num_eigs))
        for i in range(num_eigs):
            corrections[:, i] = preconditioner(residuals[:, i], current_eig_vals[i])

        # Add new correction vectors to the subspace V, then re-orthonormalize
        V = np.hstack((V, corrections))

        ## rocsolver_dgeqrf(
        V, _ = np.linalg.qr(V) # Standard QR for orthonormalization 	 	rocsolver_zgeqrf
        
        #V = corrections
        
    print("Davidson did not converge within max iterations.")
    return current_eig_vals, current_eig_vecs

# Example usage:
if __name__ == "__main__":
    # Create a large, sparse, diagonally dominant matrix
    N = 100
    np.random.seed(0)
    A = np.diag(np.arange(1, N + 1).astype(float))
    # Add some small off-diagonal elements
    A += np.random.rand(N, N) * 0.001
    A = (A + A.T) / 2 # Ensure symmetry
    
    # Compute the lowest 3 eigenvalues
    eig_vals, eig_vecs = davidson_algorithm(A, num_eigs=3)
    
    print("\nComputed eigenvalues:")
    print(eig_vals)
    
    # Verify with standard numpy solver
    true_eig_vals = np.linalg.eigvalsh(A)
    print("\nTrue lowest eigenvalues (NumPy):")
    print(true_eig_vals[:3])
