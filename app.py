import numpy as np
import pickle
from flask import Flask, request, render_template, url_for
import lime.lime_tabular
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shap
import os
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model
model = pickle.load(open("RFC_Model", "rb"))

# Feature columns
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Training data for encoding and LIME
X_train_raw = pd.read_csv("Telco-Customer-Churn.csv")
X_train_raw['TotalCharges'] = pd.to_numeric(X_train_raw['TotalCharges'], errors='coerce')

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    le.fit(X_train_raw[col].astype(str))
    label_encoders[col] = le

X_train_processed = X_train_raw[feature_names].copy()
X_train_processed[numerical_features] = X_train_processed[numerical_features].apply(pd.to_numeric, errors='coerce')
X_train_processed.dropna(inplace=True)
for col in categorical_features:
    X_train_processed[col] = label_encoders[col].transform(X_train_processed[col].astype(str))

# SHAP
shap_explainer = shap.Explainer(model)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        input_data = []
        for feature in feature_names:
            value = form_data.get(feature)
            if feature in numerical_features:
                input_data.append(float(value))
            else:
                input_data.append(label_encoders[feature].transform([value])[0])

        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Model prediction
        risk_score = model.predict_proba(input_df)[0][1]
        risk_category = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

        if risk_category == "High":
            retention_actions = ["Grant loyalty benefits", "Offer cashback offers", "Schedule agent call to customer"]
        elif risk_category == "Medium":
            retention_actions = ["Grant loyalty points"]
        else:
            retention_actions = ["No Action Required"]

        # Rule-based logic
        def rule_based_risk(data):
            high = [
                data['Contract'] == 'Month-to-month',
                data['TechSupport'] == 'No',
                data['OnlineSecurity'] == 'No',
                data['InternetService'] == 'Fiber optic',
                data['PaymentMethod'] == 'Electronic check',
                data['DeviceProtection'] == 'No',
                data['OnlineBackup'] == 'No',
                data['StreamingMovies'] == 'Yes',
                data['StreamingTV'] == 'Yes',
                float(data['tenure']) < 6,
                float(data['MonthlyCharges']) > 80,
                float(data['TotalCharges']) < 200,
            ]
            medium = [
                data['Contract'] == 'One year',
                data['TechSupport'] == 'No',
                data['OnlineSecurity'] == 'No',
                data['InternetService'] == 'DSL',
                data['DeviceProtection'] == 'No',
                6 <= float(data['tenure']) < 12,
                60 <= float(data['MonthlyCharges']) <= 80,
                200 <= float(data['TotalCharges']) < 500,
            ]
            return "High" if sum(high) >= 6 else "Medium" if sum(medium) >= 4 else "Low"

        rule_based_category = rule_based_risk(form_data)

        # LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_processed.values,
            feature_names=feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification"
        )
        explanation = explainer.explain_instance(input_df.values[0], model.predict_proba, num_features=5)
        lime_html = explanation.as_html()

        # SHAP
        try:
            shap_values = shap_explainer(input_df)
            force_html = shap.plots.force(shap_values[0], matplotlib=False)

            shap_html = f"""
            <!DOCTYPE html>
            <html>
            <head><meta charset='utf-8'><script src='https://cdn.jsdelivr.net/npm/shap@latest/dist/bundle.js'></script></head>
            <body><div id='shap'></div><script>{force_html.js_code}</script></body>
            </html>
            """
            os.makedirs("templates", exist_ok=True)
            with open("templates/shap.html", "w", encoding="utf-8") as f:
                f.write(shap_html)

            shap_path = url_for("shap_plot")
        except Exception as e:
            logger.error(f"SHAP error: {e}")
            shap_path = None

        return render_template("index.html",
                               risk_score=round(risk_score * 100, 2),
                               risk_category=risk_category,
                               retention_actions=retention_actions,
                               rule_based_category=rule_based_category,
                               lime_html=lime_html,
                               shap_plot=shap_path)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Error: {str(e)}"

@app.route("/shap")
def shap_plot():
    return render_template("shap.html")

if __name__ == "__main__":
    app.run(debug=True)
