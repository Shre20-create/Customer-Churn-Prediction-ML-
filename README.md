# Customer-Churn-Prediction-ML-
Skills: Classification, feature engineering, model evaluation Build
Project Structure
customer-churn-prediction/
│
├─ README.md
├─ requirements.txt
├─ data/
│   └─ Telco-Customer-Churn.csv
│
├─ notebooks/
│   └─ churn_notebook.ipynb
│
├─ models/
│   ├─ rf_churn_model.pkl
│   ├─ scaler.pkl
│   └─ encoder_dict.pkl
│
├─ apps/
│   └─ churn_app_force.py
│
└─ scripts/
    └─ preprocess.py
File Contents
1️⃣ README.md
# Customer Churn Prediction

## Overview
This project predicts which customers are likely to churn using machine learning models (Random Forest & XGBoost) and explains predictions using SHAP.

### Features:
- Data preprocessing and feature engineering
- Classification modeling
- Model evaluation (Accuracy, F1-Score, ROC-AUC)
- Model explainability with SHAP
- Interactive Streamlit app with live SHAP force plots

## Dataset
Telco Customer Churn dataset from Kaggle: [Link](https://www.kaggle.com/blastchar/telco-customer-churn)

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
2. Run Jupyter Notebook to train the model:
jupyter notebook notebooks/churn_notebook.ipynb
3. Run Streamlit app:
streamlit run apps/churn_app_force.py
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
streamlit
joblib
