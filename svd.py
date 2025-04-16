import numpy as np

# Original matrix A
A = np.array([[3, 2], [2, 3]])

# Apply SVD
U, S, Vt = np.linalg.svd(A)

# Reconstruct A
Sigma = np.diag(S)
A_reconstructed = U @ Sigma @ Vt

print("Original A:\n", A)
print("Reconstructed A:\n", A_reconstructed)
