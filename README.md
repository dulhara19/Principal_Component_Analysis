# 📐 Matrix Rank Calculator & Iris Dataset Feature Analysis

## 🚀 Overview

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
