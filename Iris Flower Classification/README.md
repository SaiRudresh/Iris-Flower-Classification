# Iris Flower Classification - Data Science Project

This project involves building a classification model to identify Iris flowers into three species using petal and sepal length/width features. The project includes data preprocessing, model training with Random Forest, and deployment of a prediction script.

---

## Objective
- Build a classification model to identify Iris flower species (Setosa, Versicolor, Virginica).
- Identify the most significant features influencing classification.
- Evaluate model performance using appropriate techniques.

---

## Project Structure
```
├── iris_classification.py       # Main training script
├── iris_rf_model.pkl            # Trained Random Forest model
├── iris_scaler.pkl              # Scaler used for feature normalization
├── predict_iris.py              # Predict species from new sample
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```
---

## Installation
1. Clone the repository:
```bash
https://github.com/SaiRudresh/Iris-Flower-Classification.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
---

## Running the Code
### 1. Train the Model
```bash
python iris_classification.py
```
This will create `iris_rf_model.pkl` and `iris_scaler.pkl`.

### 2. Run Prediction Script
```bash
python predict_iris.py
```
Modify the `new_data` in `predict_iris.py` with new input measurements to predict species.

--- 

## Model Evaluation
- **Accuracy:** High accuracy achieved through Random Forest.
- **Feature Importance:** Petal length and petal width are most significant.

---

## Dataset Description
The dataset includes the following features:
- **Sepal length (cm)**
- **Sepal width (cm)**
- **Petal length (cm)**
- **Petal width (cm)**

The dataset used is the **Iris dataset** from `sklearn.datasets`.

---

## Feature Importance
Based on the trained model, the most significant features for classification are:
- **Petal length (cm)**
- **Petal width (cm)**

Feature importance was extracted using `model.feature_importances_`.

---



## Tools and Libraries
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## Model Used
- Random Forest Classifier (n_estimators=100, random_state=42)

---
## License
This project is for educational purposes.

---

## Author
- Sai Rudresh Reddy Gunda

---
Happy coding!!!

