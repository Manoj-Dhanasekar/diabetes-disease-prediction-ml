import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(data_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Replace zero values with median
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols:
        data[col] = data[col].replace(0, data[col].median())

    # Split features and target
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    # Save model
    joblib.dump(model, "../models/diabetes_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")

    print("Model and Scaler saved successfully!")


if __name__ == "__main__":
    train_model("../data/diabetes.csv")