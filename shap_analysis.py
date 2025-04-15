import shap
import shap_analysis
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Convert it into a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Print the first few rows of the dataset
print(df.head())

# Split data into features (X) and target (y)
X = df.drop(columns='target')
y = df['target']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check the model's accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")



# Initialize SHAP Explainer
# explainer = shap.Explainer(model, X_train )

# Initialize SHAP Explainer with additivity check disabled
explainer = shap.Explainer(model, X_train, check_additivity=False)


# Get SHAP values for the test set
shap_values = explainer(X_test)

# SHAP waterfall plot for the first sample in the test set
shap_analysis.plots.waterfall(shap_values[0])

# SHAP summary plot for all the test set samples
shap_analysis.plots.beeswarm(shap_values)

# SHAP feature importance bar plot
shap_analysis.summary_plot(shap_values, X_test)
