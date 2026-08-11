# Customer Churn Prediction API

A machine learning API that predicts whether a telecom customer is likely to churn, built with **FastAPI** and deployed on **Render**.

🔗 **Live API:** https://churn-prediction-api-5zfe.onrender.com/docs

---

## Overview

This project trains a churn prediction model on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and serves it through a REST API. Given a customer's account and service details, the API returns:

- Whether the customer is predicted to churn (`Yes` / `No`)
- The predicted probability of churn

## Model

Four models were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Logistic Regression + SMOTE | Handles class imbalance |
| Random Forest + SMOTE | Ensemble method |
| **XGBoost + SMOTE** | **Best performer — selected as final model** |

**SMOTE** (Synthetic Minority Oversampling Technique) was used to address class imbalance in the churn dataset, since far fewer customers churn than stay.

The final model, along with the fitted scaler and the exact column order used at training time, are saved as `.pkl` files and loaded by the API at startup.

## Project Structure

── main.py # FastAPI app — loads model and serves /predict
├── churn_model.pkl # Trained XGBoost model
├── scaler.pkl # Fitted StandardScaler for numeric features
├── model_columns.pkl # Column order expected by the model
├── requirements.txt # Python dependencies
└── churn_prediction.ipynb # Notebook: data exploration, training, evaluation

## API Usage

### `GET /`
Health check — confirms the API is running.

### `POST /predict`
Takes customer details and returns a churn prediction.

**Example request body:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 450.0
}
```

**Example response:**

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.8892
}
```

## Running Locally

1. Clone the repository:it clone https://github.com/jagruthi080205/churn-prediction-api.git
cd churn-prediction-api.

2. Install dependencies:pip install -r requirements.txt

3. Start the server:uvicorn main:app --reload

4. Open the interactive docs in your browser:http://127.0.0.1:8000/docs

## Deployment

This API is deployed on [Render](https://render.com) as a free-tier Web Service, using:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

> Note: the free instance spins down after inactivity, so the first request after idle time may take up to ~50 seconds to respond while it wakes up.

## Tech Stack

- **Python**
- **FastAPI** — API framework
- **XGBoost** — final prediction model
- **scikit-learn** — preprocessing, scaling, other models
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **pandas / numpy** — data processing
- **Uvicorn** — ASGI server
- **Render** — deployment

## Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — customer account information, services subscribed, and churn label from a telecom company.
