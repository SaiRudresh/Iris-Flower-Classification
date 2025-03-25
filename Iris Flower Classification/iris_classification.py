# iris_classification.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows (optional)
print("\nSample Data:")
print(df.head())

# Basic statistics (optional)
print("\nDataset Summary:")
print(df.describe())

# Preprocessing
X = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, 'iris_rf_model.pkl')
print("\nModel saved as 'iris_rf_model.pkl'")

# Save the scaler too (optional but recommended for future predictions)
joblib.dump(scaler, 'iris_scaler.pkl')
print("Scaler saved as 'iris_scaler.pkl'")
