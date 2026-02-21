Diabetes Disease Prediction using Machine Learning
📌 Project Overview

This project builds a Machine Learning classification model to predict whether a patient is likely to have diabetes based on medical diagnostic features.

The model is developed using the Pima Indians Diabetes Dataset and includes data preprocessing, feature scaling, model training, evaluation, and prediction.

🚀 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

📊 Machine Learning Workflow

Data Cleaning & Preprocessing

Handling Zero / Missing Values

Feature Scaling using StandardScaler

Train-Test Split

Logistic Regression Model

Random Forest Classifier

Model Evaluation (Accuracy, Confusion Matrix, Classification Report)

Feature Importance Analysis

Model Saving for Deployment

📁 Project Structure
diabetes-disease-prediction-ml/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── requirements.txt
└── README.md
🎯 Model Performance

Logistic Regression Accuracy: ~75–80%

Random Forest Accuracy: ~78–85%

(Random Forest performed better compared to Logistic Regression.)

🔍 Feature Importance

Random Forest was used to analyze feature importance and identify key medical attributes influencing diabetes prediction.

Important features typically include:

Glucose Level

BMI

Age

Insulin

▶ How to Run the Project
1️⃣ Clone the repository
git clone <your-repository-link>
cd diabetes-disease-prediction-ml
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run Jupyter Notebook
jupyter notebook

Open diabetes_prediction.ipynb and run all cells.

📌 Dataset

Pima Indians Diabetes Dataset
Available on UCI Machine Learning Repository / Kaggle.

🎓 Learning Outcomes

Hands-on experience with ML classification models

Understanding data preprocessing and feature scaling

Model comparison and evaluation techniques

Saving trained ML models for deployment

Building structured ML project for GitHub portfolio

📎 Future Improvements

Hyperparameter tuning for better accuracy

Deployment using Flask or Streamlit

Integration with web-based prediction interface