# import numpy as np

# # Original matrix A
# A = np.array([[3, 2], [2, 3]])

# # Apply SVD
# U, S, Vt = np.linalg.svd(A)

# # Reconstruct A
# Sigma = np.diag(S)
# A_reconstructed = U @ Sigma @ Vt

# print("Original A:\n", A)
# print("Reconstructed A:\n", A_reconstructed)

import math
# 1. Matrix input
def get_matrix():
    print("Enter your matrix row by row, separated by commas:")
    rows = []
    while True:
        line = input("Row (or type 'done'): ")
        if line.lower() == 'done':
            break
        row = list(map(float, line.split(',')))
        rows.append(row)
    return rows

# 2. Transpose
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# 3. Matrix multiplication
def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result

# 4. Determinant (Only for 2x2)
def determinant_2x2(matrix):
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

# 5. Eigenvalues for 2x2 matrix using quadratic formula
def find_eigenvalues_2x2(A):
    a = 1
    b = -(A[0][0] + A[1][1])
    c = determinant_2x2(A)
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []
    sqrt_disc = math.sqrt(discriminant)
    l1 = (-b + sqrt_disc) / (2*a)
    l2 = (-b - sqrt_disc) / (2*a)
    return [l1, l2]

# 6. Find eigenvectors manually (limited for now)
def find_eigenvectors_2x2(A, eigenvalues):
    vectors = []
    for λ in eigenvalues:
        # Solve (A - λI)x = 0
        m = [
            [A[0][0] - λ, A[0][1]],
            [A[1][0], A[1][1] - λ]
        ]
        # Assume x2 = 1, solve for x1
        if m[0][1] != 0:
            x1 = -m[0][1]
            x2 = m[0][0]
        else:
            x1 = 1
            x2 = 0
        vectors.append([x1, x2])
    return vectors

# 7. Normalize vectors
def normalize(v):
    length = math.sqrt(sum(x**2 for x in v))
    return [x / length for x in v]

# === MAIN DRIVER ===
A = get_matrix()
print("Matrix A:", A)

At = transpose(A)
print("Transpose A:", At)

AtA = matmul(At, A)
print("A^T A:", AtA)

eigenvalues = find_eigenvalues_2x2(AtA)
print("Eigenvalues of A^T A:", eigenvalues)

eigenvectors = find_eigenvectors_2x2(AtA, eigenvalues)
V = [normalize(v) for v in eigenvectors]

print("---------------------------------")
print("Right singular vectors (V):", V)

# we can continue from here to compute Σ and U as next steps...

# 8. Create Σ matrix (diagonal matrix of singular values)
singular_values = [math.sqrt(val) for val in eigenvalues]
Σ = [
    [singular_values[0], 0],
    [0, singular_values[1]]
]
print("---------------------------------")
print("Singular values (Σ):", Σ)

# 9. Compute U = A * V / σ
U = []
for i in range(len(V)):
    Av = [
        sum(A[row][k] * V[i][k] for k in range(len(V[0])))
        for row in range(len(A))
    ]
    if singular_values[i] != 0:
        ui = [x / singular_values[i] for x in Av]
    else:
        ui = Av
    U.append(normalize(ui))
U = list(map(list, zip(*U)))  # Transpose to match usual U format

print("---------------------------------")
print("Left singular vectors (U):", U)


VT = transpose(V)
print("V^T:", VT)

# U * Σ
UΣ = matmul(U, Σ)
# A' = UΣ * V^T
A_reconstructed = matmul(UΣ, VT)
print("Reconstructed A:", A_reconstructed)
