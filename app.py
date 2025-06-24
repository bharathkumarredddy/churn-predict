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
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Feature definitions
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
    'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
    'StreamingMovies', 'StreamingTV'
]
feature_names = numerical_features + categorical_features

# Load training data for LIME and encoders
try:
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
except Exception as e:
    logger.error(f"Data loading error: {e}")
    raise

# Initialize SHAP explainer
explainer = None

def get_shap_explainer():
    global explainer
    if explainer is None:
        explainer = shap.Explainer(model, X_train_processed)
    return explainer

# Rule-based risk function
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

# SHAP visualization
def generate_shap_plot(input_df):
    try:
        explainer = get_shap_explainer()
        shap_values = explainer(input_df)
        
        base_value = shap_values.base_values[0]
        shap_values_instance = shap_values.values[0]
        features = input_df.iloc[0]
        
        force_plot = shap.plots.force(
            base_value=base_value,
            shap_values=shap_values_instance,
            features=features,
            matplotlib=False,
            show=False
        )
        
        shap_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            {shap.getjs()}
        </head>
        <body>
            {force_plot.html()}
        </body>
        </html>
        """
        
        os.makedirs("templates/shap_plots", exist_ok=True)
        plot_file = f"shap_plots/plot_{np.random.randint(10000)}.html"
        
        with open(f"templates/{plot_file}", "w", encoding="utf-8") as f:
            f.write(shap_html)
            
        return plot_file
        
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return None

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/shap_plot/<path:filename>")
def shap_plot(filename):
    return render_template(f"shap_plots/{filename}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        risk_prob = model.predict_proba(input_df)[0][1]
        risk_score = round(risk_prob * 100, 2)
        risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.4 else "Low"

        # LIME Explanation
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

        # SHAP Force Plot
        shap_plot_path = generate_shap_plot(input_df)

        # Retention Actions
        retention_actions = []
        if risk_prob > 0.7:
            retention_actions = ["Offer discount", "Assign account manager", "Free service upgrade"]
        elif risk_prob > 0.4:
            retention_actions = ["Loyalty points bonus", "Personalized email campaign"]
        else:
            retention_actions = ["Standard engagement"]

        if form_data.get('TechSupport') == 'No':
            retention_actions.append("Offer free tech support trial")

        return render_template(
            "index.html",
            risk_score=risk_score,
            risk_category=risk_category,
            retention_actions=retention_actions,
            rule_based_category=rule_based_risk(form_data),
            lime_html=lime_html,
            shap_plot_path=shap_plot_path
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)