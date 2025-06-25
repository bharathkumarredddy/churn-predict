import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import lime.lime_tabular
import shap
from sklearn.preprocessing import LabelEncoder
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = pickle.load(open("RFC_Model", "rb"))

# Define features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Prepare training data for encoders and LIME
df_train = pd.read_csv("Telco-Customer-Churn.csv")
df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'], errors='coerce')
df_train.dropna(inplace=True)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    le.fit(df_train[col].astype(str))
    label_encoders[col] = le
    df_train[col] = le.transform(df_train[col].astype(str))

df_train[numerical_features] = df_train[numerical_features].apply(pd.to_numeric, errors='coerce')
X_train = df_train[feature_names]

# SHAP Explainer
shap_explainer = shap.TreeExplainer(model)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        input_data = []
        for feature in feature_names:
            val = form_data.get(feature)
            if feature in numerical_features:
                input_data.append(float(val))
            else:
                input_data.append(label_encoders[feature].transform([val])[0])
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

        # LIME Explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification"
        )
        explanation = explainer.explain_instance(input_df.values[0], model.predict_proba, num_features=5)
        lime_html = explanation.as_html()

        # SHAP Explanation - write HTML to static file
        try:
            shap_values = shap_explainer.shap_values(input_df)
            shap_html = shap.plots.force(
                base_value=shap_explainer.expected_value[1],
                shap_values=shap_values[1][0],
                features=input_df.iloc[0],
                matplotlib=False
            )
            shap_html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset='utf-8'>
                    <script src='https://cdn.jsdelivr.net/npm/shap@latest/dist/bundle.js'></script>
                </head>
                <body>
                    <div id='shap'></div>
                    <script>{shap_html.js_code}</script>
                </body>
                </html>
            """
            os.makedirs("static", exist_ok=True)
            with open("static/shap.html", "w", encoding="utf-8") as f:
                f.write(shap_html_content)
            shap_path = url_for("static", filename="shap.html")
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
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
