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

🧑‍💻Author
Dulhara Lakshan