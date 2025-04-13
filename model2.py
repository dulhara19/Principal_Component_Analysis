import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Load Iris dataset
df = sns.load_dataset("iris")

# Encode target
df["species_encoded"] = df["species"].astype("category").cat.codes

# Select only numeric features
features = df.drop(columns=["species"])
X = features.drop(columns=["species_encoded"])
y = features["species_encoded"]

# 1️⃣ Pearson, Spearman, Kendall correlation matrices
pearson_corr = X.corr(method='pearson')
spearman_corr = X.corr(method='spearman')
kendall_corr = X.corr(method='kendall')

# 2️⃣ Mutual Information scores with the target
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores_df = pd.Series(mi_scores, index=X.columns)

# 🔍 Display the correlation matrices
print("📊 Pearson Correlation Matrix:\n", pearson_corr)
print("\n📊 Spearman Correlation Matrix:\n", spearman_corr)
print("\n📊 Kendall Correlation Matrix:\n", kendall_corr)

# 🔍 Display MI scores
print("\n🔍 Mutual Information with 'species':")
print(mi_scores_df.sort_values(ascending=False))

# Optional: Plot MI for better visualization
mi_scores_df.sort_values().plot(kind='barh', title="Mutual Information Scores with 'species'")
plt.xlabel("MI Score")
plt.tight_layout()
plt.show()
