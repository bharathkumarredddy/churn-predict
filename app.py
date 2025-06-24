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
import logging
import os
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Load model
try:
    with open("RFC_Model", "rb") as model_file:
        model = pickle.load(model_file)
    logger.info("Model loaded")
except Exception as e:
    logger.error(f"Model load error: {e}")
    raise

# Features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Load training data for encoding and LIME
try:
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        label_encoders[col] = le

    X_train = df[feature_names]
except Exception as e:
    logger.error(f"Data loading error: {e}")
    raise

# Rule-based categorization
def rule_based_risk(data):
    score = 0
    if data.get('Contract') == 'Month-to-month': score += 3
    if data.get('TechSupport') == 'No': score += 2
    if data.get('OnlineSecurity') == 'No': score += 2
    if data.get('InternetService') == 'Fiber optic': score += 1
    if data.get('PaymentMethod') == 'Electronic check': score += 1
    try:
        tenure = float(data.get('tenure', 0))
        monthly = float(data.get('MonthlyCharges', 0))
        if tenure < 6: score += 2
        elif tenure < 12: score += 1
        if monthly > 70: score += 1
    except:
        pass
    return 'High' if score >= 6 else 'Medium' if score >= 3 else 'Low'

# SHAP plot generator
def generate_shap_plot(input_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        force_html = shap.plots.force(
            explainer.expected_value[1],  # Use for class 1
            shap_values[1][0],
            input_df.iloc[0],
            matplotlib=False,
            show=False
        )

        shap_html = f"""
        <html>
        <head>
        <meta charset='utf-8'>
        <script src='https://cdn.jsdelivr.net/npm/shap@latest/dist/shap.min.js'></script>
        </head>
        <body>{force_html.html()}</body>
        </html>
        """
        with open("templates/shap.html", "w", encoding="utf-8") as f:
            f.write(shap_html)
        return True
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return False

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/shap")
def shap_route():
    return render_template("shap.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        input_data = []

        for feature in feature_names:
            value = form.get(feature)
            if feature in numerical_features:
                input_data.append(float(value))
            else:
                input_data.append(label_encoders[feature].transform([value])[0])

        input_df = pd.DataFrame([input_data], columns=feature_names)
        risk_prob = model.predict_proba(input_df)[0][1]
        risk_score = round(risk_prob * 100, 2)
        risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.4 else "Low"

        # Retention Suggestions
        actions = []
        if risk_prob > 0.7:
            actions = ["Offer discount", "Assign account manager", "Free service upgrade"]
        elif risk_prob > 0.4:
            actions = ["Loyalty bonus", "Email follow-up"]
        else:
            actions = ["Standard engagement"]
        if form.get("TechSupport") == "No":
            actions.append("Free tech support trial")

        # LIME
        lime_html = ""
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, feature_names=feature_names, class_names=["No Churn", "Churn"], mode="classification"
            )
            explanation = explainer.explain_instance(input_df.values[0], model.predict_proba, num_features=5)
            lime_html = explanation.as_html()
        except Exception as e:
            logger.warning(f"LIME error: {e}")

        # SHAP
        shap_success = generate_shap_plot(input_df)

        return render_template(
            "index.html",
            risk_score=risk_score,
            risk_category=risk_category,
            retention_actions=actions,
            rule_based_category=rule_based_risk(form),
            lime_html=lime_html,
            shap_plot="/shap" if shap_success else None
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error="Prediction failed.")

if __name__ == "__main__":
    app.run(debug=False)
