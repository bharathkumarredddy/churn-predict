import numpy as np
import pickle
from flask import Flask, request, render_template, url_for
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shap
import os
import logging
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
model = pickle.load(open("RFC_Model", "rb"))

# Feature setup
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Load and preprocess training data
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

# SHAP explainer
shap_explainer = shap.TreeExplainer(model)

# Flask app
app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

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

        risk_score = model.predict_proba(input_df)[0][1]
        risk_category = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

        if risk_score > 0.7:
            retention_actions = ["Grant loyalty benefits", "Offer cashback offers", "Schedule agent call to customer"]
        elif risk_score > 0.3:
            retention_actions = ["Grant loyalty points"]
        else:
            retention_actions = ["No Action Required"]

        # Rule-based logic
        def rule_based_risk(form_data):
            high = [
                form_data['Contract'] == 'Month-to-month',
                form_data['TechSupport'] == 'No',
                form_data['OnlineSecurity'] == 'No',
                form_data['InternetService'] == 'Fiber optic',
                form_data['PaymentMethod'] == 'Electronic check',
                form_data['DeviceProtection'] == 'No',
                form_data['OnlineBackup'] == 'No',
                form_data['StreamingMovies'] == 'Yes',
                form_data['StreamingTV'] == 'Yes',
                float(form_data['tenure']) < 6,
                float(form_data['MonthlyCharges']) > 80,
                float(form_data['TotalCharges']) < 200,
            ]
            medium = [
                form_data['Contract'] == 'One year',
                form_data['TechSupport'] == 'No',
                form_data['OnlineSecurity'] == 'No',
                form_data['InternetService'] == 'DSL',
                form_data['DeviceProtection'] == 'No',
                6 <= float(form_data['tenure']) < 12,
                60 <= float(form_data['MonthlyCharges']) <= 80,
                200 <= float(form_data['TotalCharges']) < 500,
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

        # SHAP visualization
        try:
            shap_values = shap_explainer(input_df)
            
            # Create and save waterfall plot
            plt.figure()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            shap_path = os.path.join(app.config['STATIC_FOLDER'], 'shap_plot.png')
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            shap_url = url_for('static', filename='shap_plot.png')
        except Exception as e:
            logger.error(f"SHAP error: {e}")
            shap_url = None

        return render_template("index.html",
                            risk_score=round(risk_score * 100, 2),
                            risk_category=risk_category,
                            retention_actions=retention_actions,
                            rule_based_category=rule_based_category,
                            lime_html=lime_html,
                            shap_plot=shap_url)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True)