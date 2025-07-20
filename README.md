# Customer Churn Prediction Project

## Project Goal

The primary goal of this project is to build and deploy a machine learning model that accurately predicts customer churn for a fictional subscription-based telecommunications company.

By identifying customers who are likely to cancel their service, the business can proactively engage them with targeted retention strategies—such as special offers or improved customer support—ultimately reducing revenue loss.

This project serves as an end-to-end demonstration of a standard data science workflow, from **data acquisition and preprocessing** to **model training, evaluation, and deployment** via a simple web application.

---

## Project Structure

```
.
├── models/
│   └── churn_logistic_regression.joblib
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   └── predict.py
├── templates/
│   ├── index.html
│   └── result.html
├── app.py
├── requirements.txt
└── README.md
```

- **`models/`**: Stores the trained and serialized machine learning model (`.joblib` file).
- **`src/`**: Contains the core Python source code.
  - `config.py`: Centralizes configuration variables like file paths and URLs.
  - `train.py`: Downloads data, preprocesses it, trains the model, and saves the final pipeline.
  - `predict.py`: Utility script to make predictions from the command line on a sample customer.
- **`templates/`**: Holds the HTML files for the web user interface.
- **`app.py`**: The main Flask application that serves the UI and handles prediction requests.
- **`requirements.txt`**: Lists all Python libraries required to run the project.
- **`README.md`**: This file, providing a complete overview of the project.

---

## Dataset

This project uses the **Telco Customer Churn dataset**, originally provided by IBM. It contains **7,043 records**, where each record represents a unique customer.

### Features:

- **Customer Services**: 
  - `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
- **Customer Account Information**: 
  - `Contract`, `PaymentMethod`, `tenure`, `MonthlyCharges`
- **Customer Demographics**: 
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Target Variable**: 
  - `Churn` – Indicates whether the customer left the company within the last month.

---

## Model Details

The predictive model is a **Logistic Regression classifier** wrapped in a **Scikit-learn Pipeline**.

### Pipeline Stages:

1. **Preprocessor (`ColumnTransformer`)**:
   - **Numerical Features**: Scaled using `StandardScaler` (zero mean, unit variance).
   - **Categorical Features**: Transformed using `OneHotEncoder`.

2. **Classifier (`LogisticRegression`)**:
   - Learns the relationship between the preprocessed features and churn probability.

---

## Technology Stack

- **Backend**: Python
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Web Framework**: Flask
- **Model Serialization**: Joblib

---

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt
```

---

## How to Run

### 1. Train the Model

Run the training script from the project's root directory:

```bash
python src/train.py
```

This script will:
- Download the dataset
- Preprocess it
- Train the logistic regression model
- Evaluate performance
- Save the trained pipeline in the `models/` directory

---

### 2. Launch the Web Application

Once the model is trained, start the Flask web app:

```bash
python app.py
```

Then navigate to:

```
http://127.0.0.1:5000
```

You’ll see a web form where you can input customer data and get a real-time churn prediction.

---

## End-to-End Workflow Covered

- Data ingestion  
- Cleaning and preprocessing  
- Feature engineering  
- Model training and evaluation  
- Deployment via Flask  
- Interactive web UI

---

## Notes

- The pipeline ensures that **training and inference use identical preprocessing steps**.
- This setup is modular and easily extendable to other ML models or frameworks (like XGBoost or FastAPI).

---

> This is a simplified real-world use case, perfect for learning how to build and ship ML models into production environments.
