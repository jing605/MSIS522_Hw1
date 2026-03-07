# MSIS 522 HW1 - Titanic Survival Prediction

This project completes the full data science workflow for Homework 1 in MSIS 522.  
The dataset used is the Titanic dataset, and the task is a binary classification problem predicting whether a passenger survived.

## Project Contents

- **msis522_hw1.ipynb**: Full exploratory data analysis, model training, evaluation, and SHAP explainability
- **app.py**: Streamlit application
- **data/**: Titanic dataset
- **models/**: Saved trained models
- **outputs/**: Model comparison results
- **requirements.txt**: Required Python packages

## Models Implemented

- Logistic Regression
- Decision Tree (with GridSearchCV)
- Random Forest (with GridSearchCV)
- XGBoost (with GridSearchCV)
- Multi-Layer Perceptron (MLP)

## Explainability

SHAP analysis is performed on the best-performing tree-based model to understand feature importance and prediction drivers.

## Streamlit App

The app includes:
- Executive Summary
- Descriptive Analytics
- Model Performance
- Explainability & Interactive Prediction

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
