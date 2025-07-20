# src/train.py

# 1. IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Import configuration
from config import DATA_URL, MODEL_PATH

def train_model():
    """
    This function handles the end-to-end process of training the churn prediction model.
    """
    # 2. LOADING THE DATA
    print("Loading data...")
    df = pd.read_csv(DATA_URL)

    # 3. DATA PREPROCESSING
    print("Preprocessing data...")
    # Handle missing values in 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Convert target variable 'Churn' to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Define Features (X) and Target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 4. SETTING UP PREPROCESSING PIPELINES
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. SPLITTING THE DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. DEFINING THE MODEL PIPELINE
    log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', log_reg_model)])

    # 7. TRAINING THE MODEL
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete!")

    # 8. EVALUATING THE MODEL
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # 9. SAVING THE MODEL
    print(f"\nSaving model to {MODEL_PATH}...")
    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print("Model saved successfully!")

if __name__ == '__main__':
    train_model()
