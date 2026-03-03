import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import gradio
from dotenv import load_dotenv
from openai import OpenAI
from prompt import create_prompt

load_dotenv()

# Load artifacts
pipeline = joblib.load("artifacts/rf_smote_pipeline.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")
metrics = joblib.load("artifacts/metrics.pkl")
target_mapping = joblib.load("artifacts/target_mapping.pkl")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("INCEPTION_API_KEY"), base_url="https://api.inceptionlabs.ai/v1"
)

# Manual mappings for categorical features (to avoid encoder issues)
# These must match exactly what was used during training

# Sex mapping
sex_mapping = {"female": 0, "male": 1}

# Housing mapping
housing_mapping = {"free": 0, "own": 1, "rent": 2}

# Saving accounts mapping
saving_mapping = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3, "none": 4}

# Checking account mapping
checking_mapping = {"little": 0, "moderate": 1, "rich": 2, "none": 3}

# Purpose mapping (manual encoding - must match training order)
purpose_mapping = {
    "car": 0,
    "furniture/equipment": 1,
    "radio/TV": 2,
    "domestic appliances": 3,
    "repairs": 4,
    "education": 5,
    "business": 6,
    "vacation/others": 7,
}

# Feature options for Gradio dropdowns
purpose_options = list(purpose_mapping.keys())
job_options = [0, 1, 2, 3]

# Column order expected by the model (from training)
FEATURE_ORDER = [
    "age",
    "sex",
    "job",
    "housing",
    "saving_accounts",
    "checking_account",
    "credit_amount",
    "credit_per_month",
    "duration",
]


# Prediction
def predict_and_recommend(
    age, sex, job, housing, savings, checking, credit_amount, duration
):
    """Predict credit risk and get LLM recommendation"""

    # Calculate credit_per_month (derived feature from training)
    credit_per_month = credit_amount / duration if duration > 0 else 0

    # Encode categorical features using manual mappings (NO purpose - not used in training)
    features = {
        "age": age,
        "sex": sex_mapping.get(sex, 0),
        "job": job,
        "housing": housing_mapping.get(housing, 0),
        "saving_accounts": saving_mapping.get(savings, 0),
        "checking_account": checking_mapping.get(checking, 0),
        "credit_amount": credit_amount,
        "credit_per_month": credit_per_month,
        "duration": duration,
    }

    # Ensure correct feature order (must match training order)
    feature_order = [
        "age",
        "sex",
        "job",
        "housing",
        "saving_accounts",
        "checking_account",
        "credit_amount",
        "credit_per_month",
        "duration",
    ]
    df = pd.DataFrame([features])[feature_order]

    # Model prediction
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    risk_label = "HIGH RISK (Bad)" if prediction == 1 else "LOW RISK (Good)"
    confidence = probability if prediction == 1 else 1 - probability

    # Get recommendation from LLM
    prompt = create_prompt(
        age,
        sex,
        job,
        housing,
        savings,
        checking,
        credit_amount,
        duration,
        "various",  # purpose - not used in model
        risk_label,
        f"{confidence:.1%}",
    )

    response = client.chat.completions.create(
        model="mercury-2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    recommendation = response.choices[0].message.content

    return risk_label, f"{confidence:.1%}", recommendation


def get_metrics():
    """Return metrics for Model Showcase tab"""
    return metrics


def get_feature_names():
    """Return feature names"""
    return feature_names
