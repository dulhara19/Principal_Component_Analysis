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
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset
data = sns.load_dataset('iris')

print(data.head())
# print(features.head())

# Get unique values in the species column
unique_species = data['species'].unique()
num_unique_species = data['species'].nunique()

print("Unique values in 'species':", unique_species)
print("Number of unique species:", num_unique_species)

from sklearn.preprocessing import LabelEncoder

# Make a copy of the dataset
encoded_data = data.copy()

# Encode the species column
le = LabelEncoder()
encoded_data['species_encoded'] = le.fit_transform(data['species'])
print("Encoded species column:")
# print(encoded_data.head(10))
# Drop the original species column
features = encoded_data.drop(columns=['species_encoded','species'])
print(features.head())

# Step 1: Calculate the correlation matrix
corr_matrix = features.corr()

# Step 2: Calculate the rank of the correlation matrix using numpy
rank = np.linalg.matrix_rank(corr_matrix.values)
print("Rank of the correlation matrix:", rank)

# Set a threshold to identify strong correlations
threshold = 0.9
high_corr_pairs = []

# Step 3: Find highly correlated feature pairs (excluding the diagonal)
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))

# Step 4: Display the results
print("\nHighly correlated feature pairs (|correlation| > {}):".format(threshold))
for f1, f2, corr in high_corr_pairs:
    print(f"{f1} ↔ {f2} → Correlation: {corr:.2f}")

#----------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

 
# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 2: Apply PCA
pca = PCA(n_components=4)  # Reduce to 2 dimensions
pca_components = pca.fit_transform(scaled_features)

# Step 3: Create a new DataFrame for plotting
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2','PC3','PC4'])
pca_df['species'] = data['species']

# Step 4: Plot the results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1',y='PC3',hue='species', palette='Set1', s=100)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.grid(True)
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained by the first two components:", sum(pca.explained_variance_ratio_))