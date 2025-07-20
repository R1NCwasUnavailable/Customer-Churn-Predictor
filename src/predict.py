# src/predict.py

import pandas as pd
import joblib
import argparse

# Import configuration
from config import MODEL_PATH

def predict_churn(customer_data):
    """
    Loads the trained model and predicts churn for a given customer.

    Args:
        customer_data (pd.DataFrame): A DataFrame containing the new customer's data.

    Returns:
        A tuple containing the prediction (0 or 1) and the probability of churn.
    """
    try:
        # Load the trained model pipeline
        model_pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return None, None

    # Make predictions
    prediction = model_pipeline.predict(customer_data)[0]
    probability = model_pipeline.predict_proba(customer_data)[0][1] # Probability of class 1 (Churn)

    return prediction, probability

if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    # This allows a developer to pass in customer data from the terminal for quick checks.
    parser = argparse.ArgumentParser(description="Predict customer churn for a given customer.")
    
    # Add arguments for each feature
    parser.add_argument("--gender", type=str, default="Female")
    parser.add_argument("--SeniorCitizen", type=int, default=0)
    parser.add_argument("--Partner", type=str, default="Yes")
    parser.add_argument("--Dependents", type=str, default="No")
    parser.add_argument("--tenure", type=int, default=1)
    parser.add_argument("--PhoneService", type=str, default="No")
    parser.add_argument("--MultipleLines", type=str, default="No phone service")
    parser.add_argument("--InternetService", type=str, default="DSL")
    parser.add_argument("--OnlineSecurity", type=str, default="No")
    parser.add_argument("--OnlineBackup", type=str, default="Yes")
    parser.add_argument("--DeviceProtection", type=str, default="No")
    parser.add_argument("--TechSupport", type=str, default="No")
    parser.add_argument("--StreamingTV", type=str, default="No")
    parser.add_argument("--StreamingMovies", type=str, default="No")
    parser.add_argument("--Contract", type=str, default="Month-to-month")
    parser.add_argument("--PaperlessBilling", type=str, default="Yes")
    parser.add_argument("--PaymentMethod", type=str, default="Electronic check")
    parser.add_argument("--MonthlyCharges", type=float, default=29.85)
    parser.add_argument("--TotalCharges", type=float, default=29.85)

    args = parser.parse_args()

    # Create a DataFrame from the command-line arguments
    customer_data_dict = vars(args)
    customer_df = pd.DataFrame([customer_data_dict])

    print("--- Predicting for Customer with the following data ---")
    print(customer_df.T)

    prediction, probability = predict_churn(customer_df)

    if prediction is not None:
        churn_status = "CHURN" if prediction == 1 else "NOT CHURN"
        print(f"\nPrediction: Customer will {churn_status}")
        print(f"Probability of Churn: {probability:.2%}")

