# app.py

from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# Import configuration
from src.config import MODEL_PATH

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model pipeline
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
    model_pipeline = None

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request from the form."""
    if model_pipeline is None:
        return "Model not loaded. Please train the model first by running src/train.py.", 500

    # Get data from the form
    form_data = request.form.to_dict()

    # Convert form data to a pandas DataFrame
    # Important: The column names must match the training data
    try:
        # Convert numerical fields from string to number
        form_data['SeniorCitizen'] = int(form_data['SeniorCitizen'])
        form_data['tenure'] = int(form_data['tenure'])
        form_data['MonthlyCharges'] = float(form_data['MonthlyCharges'])
        form_data['TotalCharges'] = float(form_data['TotalCharges'])
        
        customer_df = pd.DataFrame([form_data])
    except (ValueError, KeyError) as e:
        return f"Error processing form data: {e}. Please ensure all fields are filled correctly.", 400


    # Make prediction
    prediction = model_pipeline.predict(customer_df)[0]
    probability = model_pipeline.predict_proba(customer_df)[0][1] # Probability of Churn

    # Determine the result text
    churn_status = "CHURN" if prediction == 1 else "NOT CHURN"

    return render_template('result.html', status=churn_status, probability=f"{probability:.2%}")

if __name__ == '__main__':
    # Use os.environ.get('PORT', 5000) for compatibility with deployment platforms like Heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
