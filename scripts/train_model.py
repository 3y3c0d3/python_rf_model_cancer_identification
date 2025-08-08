import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import joblib
import os

# Load the dataset
cancer = load_breast_cancer()

# Create a pandas DataFrame for easier manipulation
# The data itself is in 'cancer.data', and column names are in 'cancer.feature_names'
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

# Add the target variable (what we want to predict) to the DataFrame
# 0 = malignant, 1 = benign
df['target'] = cancer.target

# --- Let's perform some basic Exploratory Data Analysis (EDA) ---

# Display the first 5 rows of the data
print("--- First 5 Rows ---")
print(df.head())

# Get a concise summary of the DataFrame
print("\n--- Data Info ---")
df.info()

# Check the distribution of the target variable
print("\n--- Target Distribution ---")
print(df['target'].value_counts())
# This shows us how many benign (1) and malignant (0) samples we have.

# Separate features (X) and target (y)
X = df.drop('target', axis=1) # All columns except the target
y = df['target'] # Only the target column

# Split the data into training and testing sets
# We'll use 70% of the data for training and 30% for testing.
# 'random_state=42' ensures that we get the same split every time we run the code,
# making our results reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create an instance of the Random Forest Classifier
# n_estimators=100 means the forest will be composed of 100 decision trees.
# random_state=42 ensures the model's results are reproducible.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
# This is where the model "learns" the patterns.
rf_model.fit(X_train, y_train)

print("\nRandom Forest model has been trained successfully!")

# Use the trained model to make predictions on the test set
y_pred = rf_model.predict(X_test)

print("\nPredictions made on the test set.")

# 1. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Accuracy ---")
print(f"The model is {accuracy:.2%} accurate.")

# 2. Confusion Matrix
# This shows us where the model got things right and where it got them wrong.
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# For a prettier view of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 3. Classification Report
# This gives us precision, recall, and f1-score for each class.
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features ---")
print(feature_importance_df.head(10))

# Visualize the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
