# 📐 Matrix Rank Calculator & Iris Dataset Feature Analysis

## Overview

This project contains two parts:

1. **Matrix Rank Calculator from Scratch**  
   A Python program that computes the **rank of any given matrix** without using NumPy’s built-in functions.

2. **Iris Dataset Feature Correlation Analysis**  
   A data analysis pipeline that identifies **highly correlated features** using matrix rank, and introduces the basics of **PCA (Principal Component Analysis)**.

--- 
## 📘 Part 1: Matrix Rank Calculator

### 🧠 What is Matrix Rank?

The **rank** of a matrix is the number of **linearly independent rows or columns**. It gives deep insight into the structure of the matrix and helps in solving equations, detecting redundancy, and more.

### ⚙️ What the Code Does

- Takes a matrix (as a list of lists).
- Applies **Gaussian elimination (row operations)** to convert it into **row echelon form**.
- Counts the number of **non-zero rows** → That is the **rank**!

### ✅ Features

- Works on any **M × N** matrix.
- Built from scratch (no NumPy used for rank).
- Helpful for educational purposes and understanding linear algebra.

---

## 📘 Part 2: Iris Dataset Analysis Using Rank

### 🌸 Dataset Used
- [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) from Seaborn
- Contains 4 numerical features and a target species column.

### 🔍 What We Did

- Loaded the dataset and dropped the label column (`species`).
- Converted features into a matrix.
- Calculated the **rank of the feature matrix** to understand how many features are linearly independent.
- Identified if any features were redundant or highly correlated.
- Discussed how this helps in **dimensionality reduction**.

### 🧠 Bonus: PCA Introduction
- Introduced the idea of **PCA (Principal Component Analysis)**.
- Explained how **eigenvectors and eigenvalues** from the covariance matrix help us reduce features while preserving data variance.

---

## 📦 Libraries Used

```bash
pip install pandas seaborn matplotlib scikit-learn
```
```py
Original Matrix:
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

Reduced Row Echelon Form:
[[1.0, 0.0, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]

Matrix Rank: 2

Iris Feature Matrix Rank: 4
Conclusion: All features are linearly independent in Iris dataset.
```

📈 Why This Project?

Solidifies understanding of linear algebra concepts.
Shows how matrix rank is useful in data science and ML preprocessing.
Builds foundation to implement PCA from scratch.

---

## 🧠 What is PCA (Principal Component Analysis)?

**Principal Component Analysis (PCA)** is a powerful technique used in **machine learning and data science** to simplify large datasets by reducing their dimensions while keeping the most important information.

---

### 📉 Why Use PCA?

- ✅ To **reduce the number of features** in a dataset (dimensionality reduction)
- ✅ To **remove noise or redundancy**
- ✅ To **visualize high-dimensional data** in 2D or 3D
- ✅ To **speed up training** of ML models
---

### ⚙️ How PCA Works (Simple Explanation)

1. **Standardize the dataset**  
   Make sure features are on the same scale (mean = 0, variance = 1)

2. **Compute the Covariance Matrix**  
   This shows how features vary together

3. **Find Eigenvectors and Eigenvalues**  
   These represent the **principal components** (directions of maximum variance)

4. **Sort and Select Top Components**  
   Choose the top `k` eigenvectors (with highest eigenvalues)

5. **Transform the Data**  
   Project original data onto these new directions → reduced data
---

### 🔢 Example

If your dataset has 5 features, and you apply PCA and keep only 2 components:
- You now have a 2D representation of your data
- Most of the **original information (variance)** is still preserved
---

### 📊 PCA and Matrix Rank

- **Matrix rank** tells you how many **independent dimensions** the data really has.
- If some features are **linearly dependent**, PCA will **eliminate** them by reducing dimensions.

---
### 📦 PCA in Practice (Scikit-learn Example)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Load Iris dataset
iris = sns.load_dataset("iris")
X = iris.drop("species", axis=1)

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Result is a new 2D dataset
print(X_pca[:5])
```

## Here are a few scenarios where you might want to reconsider using PCA

1. Non-linear relationships between features

    PCA is linear in nature, meaning it assumes that the relationships between features are linear. If the data has non-linear relationships (e.g., in datasets where features interact in complex ways), PCA may not capture the underlying structure effectively.

    Alternative: You might consider using Kernel PCA or other non-linear dimensionality reduction techniques like t-SNE or UMAP for non-linear datasets.

2. Presence of outliers

    PCA is sensitive to outliers, as it tries to maximize variance. If your data contains outliers, PCA may give disproportionate importance to these points, which could skew the results.

    Alternative: You could use robust methods like Robust PCA or preprocess the data to handle outliers (e.g., by removing them or using robust scaling techniques).

3. Categorical data

    PCA works best with continuous numerical data. If your dataset contains mostly categorical features, PCA may not be suitable. Categorical data doesn't have a natural order, so applying PCA directly may not make sense.

    Alternative: If you need to work with categorical data, you might use techniques like One-Hot Encoding or Factor Analysis and then apply PCA on the resulting features. Alternatively, you could use Multiple Correspondence Analysis (MCA).

4. When interpretability is critical

    One of the main criticisms of PCA is that it produces combinations of features (principal components) that are often hard to interpret. If you need to keep the features understandable and meaningful, the resulting principal components may be difficult to explain.

    Alternative: You might use simpler dimensionality reduction methods like Linear Discriminant Analysis (LDA) or select a subset of features based on domain knowledge.

5. Assumptions about the data distribution

    PCA assumes that the data is normally distributed and that the variance of the features reflects their importance. If your data is not normally distributed or has skewed distributions, PCA may not be the best tool for reducing dimensionality.

    Alternative: Independent Component Analysis (ICA) or Factor Analysis may work better for data that doesn't adhere to the assumptions of PCA.

6. High correlation between components

    While PCA is effective at reducing dimensions by identifying orthogonal components, it may struggle when the components are highly correlated or the variance across components is similar. This could lead to retaining too many components without significantly improving data representation.

    Alternative: Sparse PCA or Non-negative Matrix Factorization (NMF) might handle such cases better.

7. Data with different units or scales

    PCA is sensitive to the scale of the data. If your features have different units (e.g., height in cm and weight in kg), PCA might give more importance to features with larger ranges or variances.

    Solution: It’s essential to normalize or standardize the data (e.g., using Z-score normalization) before applying PCA to make sure each feature contributes equally.

8. Small datasets

    If you have a small dataset with a high number of features, PCA may not be effective since there isn't enough data to capture meaningful patterns and relationships in the features.

    Alternative: You could use regularization techniques (like Lasso or Ridge regression) for feature selection or try simpler models like decision trees, which are less affected by dimensionality.

### In Summary:

PCA is a powerful tool for dimensionality reduction, but it’s important to consider the nature of your data and your analysis goals before applying it. In cases of non-linearity, outliers, or categorical data, alternative methods may provide better results. Always remember to preprocess your data accordingly and ensure that the assumptions behind PCA align with your data characteristics.

🧑‍💻Author-
Dulhara Lakshan :)  