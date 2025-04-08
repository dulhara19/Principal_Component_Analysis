def rank_of_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    rank = 0

    for r in range(min(rows, cols)):
        # Step 1: Find a pivot
        if matrix[r][r] == 0:
            for i in range(r+1, rows):
                if matrix[i][r] != 0:
                    matrix[r], matrix[i] = matrix[i], matrix[r]  # Swap
                    break
        if matrix[r][r] == 0:
            continue

        # Step 2: Eliminate below the pivot
        for i in range(r+1, rows):
            ratio = matrix[i][r] / matrix[r][r]
            for j in range(cols):
                matrix[i][j] -= ratio * matrix[r][j]

    # Step 3: Count non-zero rows
    for row in matrix:
        if any(abs(val) > 1e-10 for val in row):  # account for float precision
            rank += 1

    return rank


matrix1 = [
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
]

print(rank_of_matrix(matrix1))

#----------------------print------------------------------------
import pandas as pd

def find_highly_correlated_features(data, threshold=0.9):
    # Step 1: Calculate the correlation matrix
    corr_matrix = data.corr()
    
    # Step 2: Find pairs of features with high correlation
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                feature_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                correlated_features.add(feature_pair)

    return correlated_features

# Example usage:
import seaborn as sns
data = sns.load_dataset('iris')  # Iris dataset
correlated = find_highly_correlated_features(data)
print("Highly correlated features:", correlated)


