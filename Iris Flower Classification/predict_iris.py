import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('iris_rf_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Define column names matching what was used during training
column_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Create new data as a DataFrame
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=column_names)

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Predict the species
prediction = model.predict(new_data_scaled)

species = ['Setosa', 'Versicolor', 'Virginica']
print(f"Predicted species: {prediction[0]}")
