import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Load dataset
data = sns.load_dataset('iris')

# Encode the target column 'species'
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])

# Drop original species column
features = data.drop(columns=['species'])
X = features.drop(columns=['species_encoded'])
y = features['species_encoded']

# Compute Mutual Information
mi_scores = mutual_info_classif(X, y, discrete_features=False)

# Display results
mi_results = pd.Series(mi_scores, index=X.columns)
mi_results = mi_results.sort_values(ascending=False)

print("\n Mutual Information Scores:")
print(mi_results)

# Optional: Plot the results
import matplotlib.pyplot as plt
mi_results.plot(kind='barh', title="Mutual Information with Target", figsize=(8,4))
plt.xlabel("MI Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
