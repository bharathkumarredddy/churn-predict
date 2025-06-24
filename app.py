import numpy as np
import pickle
from flask import Flask, request, render_template
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
import os
import sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load model and data
try:
    # Check sklearn version compatibility
    logger.info(f"Current sklearn version: {sklearn.__version__}")
    
    with open("RFC_Model", "rb") as model_file:
        model = pickle.load(model_file)
    logger.info("Model loaded successfully")
    
    # Ensure model has feature names
    if hasattr(model, 'feature_names_in_'):
        logger.info("Model has feature names")
    else:
        logger.warning("Model was fitted without feature names")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Define features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Load and preprocess training data
try:
    X_train_raw = pd.read_csv("Telco-Customer-Churn.csv")
    X_train_raw['TotalCharges'] = pd.to_numeric(X_train_raw['TotalCharges'], errors='coerce')
    
    # Initialize LabelEncoders
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        le.fit(X_train_raw[col].astype(str))
        label_encoders[col] = le
    
    # Prepare training data for LIME
    X_train_processed = X_train_raw[feature_names].copy()
    X_train_processed[numerical_features] = X_train_processed[numerical_features].apply(pd.to_numeric, errors='coerce')
    X_train_processed.dropna(inplace=True)
    for col in categorical_features:
        X_train_processed[col] = label_encoders[col].transform(X_train_processed[col].astype(str))
except Exception as e:
    logger.error(f"Data loading error: {e}")
    raise

def rule_based_risk(form_data):
    try:
        risk_factors = {
            'contract': 3 if form_data.get('Contract') == 'Month-to-month' else 0,
            'tech_support': 2 if form_data.get('TechSupport') == 'No' else 0,
            'security': 2 if form_data.get('OnlineSecurity') == 'No' else 0,
            'internet': 1 if form_data.get('InternetService') == 'Fiber optic' else 0,
            'payment': 1 if form_data.get('PaymentMethod') == 'Electronic check' else 0,
            'tenure': 2 if float(form_data.get('tenure', 0)) < 6 else 1 if float(form_data.get('tenure', 0)) < 12 else 0,
            'monthly': 1 if float(form_data.get('MonthlyCharges', 0)) > 70 else 0
        }
        total = sum(risk_factors.values())
        return "High" if total >= 6 else "Medium" if total >= 3 else "Low"
    except Exception as e:
        logger.error(f"Rule-based error: {e}")
        return "Unknown"

def generate_shap_plot(input_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Handle SHAP values based on model type
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_values = shap_values[1]
                expected_value = explainer.expected_value[1]
            else:  # Multiclass
                expected_value = explainer.expected_value[0]
                shap_values = shap_values[0]
        else:  # Regression
            expected_value = explainer.expected_value
        
        # Create the force plot
        plt.figure(figsize=(10, 5))
        shap.plots.force(
            base_value=expected_value,
            shap_values=shap_values[0],
            features=input_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Prepare input data
        form_data = request.form
        input_data = []
        
        for feature in feature_names:
            value = form_data.get(feature, '')
            if feature in numerical_features:
                try:
                    input_data.append(float(value))
                except ValueError:
                    return render_template("index.html", error=f"Invalid value for {feature}")
            else:
                try:
                    input_data.append(label_encoders[feature].transform([value])[0])
                except ValueError:
                    return render_template("index.html", error=f"Invalid value for {feature}")
        
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        try:
            risk_prob = model.predict_proba(input_df)[0][1]
            risk_score = round(risk_prob * 100, 2)
            risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.4 else "Low"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template("index.html", error="Prediction failed")

        # Generate explanations
        lime_html = None
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_processed.values,
                feature_names=feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification'
            )
            lime_exp = explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=5
            )
            lime_html = lime_exp.as_html()
        except Exception as e:
            logger.error(f"LIME error: {e}")

        # Generate SHAP plot
        shap_plot = generate_shap_plot(input_df)
        
        # Determine retention actions
        retention_actions = []
        if risk_prob > 0.7:
            retention_actions = ["Offer discount", "Assign account manager", "Free service upgrade"]
        elif risk_prob > 0.4:
            retention_actions = ["Loyalty points bonus", "Personalized email campaign"]
        else:
            retention_actions = ["Standard engagement"]
        
        # Add service-specific actions
        if form_data.get('TechSupport') == 'No':
            retention_actions.append("Offer free tech support trial")
        
        return render_template(
            "index.html",
            risk_score=risk_score,
            risk_category=risk_category,
            retention_actions=retention_actions,
            rule_based_category=rule_based_risk(form_data),
            lime_html=lime_html,
            shap_plot=shap_plot
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)