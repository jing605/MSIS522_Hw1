import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")


# -----------------------------
# Load + preprocess
# -----------------------------
@st.cache_data
def load_raw_data():
    return pd.read_csv("Titanic-Dataset.csv")


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df_model = df.copy()

    df_model["Age"] = df_model["Age"].fillna(df_model["Age"].median())
    df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])

    df_model.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
        inplace=True,
    )

    df_model = pd.get_dummies(df_model, drop_first=True)

    X = df_model.drop("Survived", axis=1)
    y = df_model["Survived"]

    return df_model, X, y


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "Neural Network (MLP)": joblib.load("mlp_model.pkl"),
    }


@st.cache_data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


@st.cache_data
def evaluate_models(X_test, y_test):
    models = load_models()

    rows = []
    roc_dict = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_dict[model_name] = (fpr, tpr, auc)
        else:
            y_prob = None
            auc = np.nan

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "AUC": auc,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    return results_df, roc_dict


def build_input_df(
    X_columns,
    pclass,
    age,
    sibsp,
    parch,
    fare,
    sex,
    embarked,
):
    input_dict = {col: 0 for col in X_columns}

    if "Pclass" in input_dict:
        input_dict["Pclass"] = pclass
    if "Age" in input_dict:
        input_dict["Age"] = age
    if "SibSp" in input_dict:
        input_dict["SibSp"] = sibsp
    if "Parch" in input_dict:
        input_dict["Parch"] = parch
    if "Fare" in input_dict:
        input_dict["Fare"] = fare

    if "Sex_male" in input_dict:
        input_dict["Sex_male"] = 1 if sex == "male" else 0

    # get_dummies(drop_first=True): C is baseline
    if "Embarked_Q" in input_dict:
        input_dict["Embarked_Q"] = 1 if embarked == "Q" else 0
    if "Embarked_S" in input_dict:
        input_dict["Embarked_S"] = 1 if embarked == "S" else 0

    return pd.DataFrame([input_dict])


# -----------------------------
# Data objects
# -----------------------------
df = load_raw_data()
df_model, X, y = preprocess_data(df)
models = load_models()
X_train, X_test, y_train, y_test = split_data(X, y)
results_df, roc_dict = evaluate_models(X_test, y_test)

best_model_name = results_df.iloc[0]["Model"]
best_tree_model_name = "XGBoost" if "XGBoost" in models else "Random Forest"
best_tree_model = models[best_tree_model_name]

# -----------------------------
# Hyperparameter table
# Replace these with your actual best params if different
# -----------------------------
hyperparams_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "Neural Network (MLP)",
        ],
        "Best Hyperparameters": [
            "max_iter=1000, random_state=42",
            "max_depth=5, min_samples_leaf=10",
            "n_estimators=200, max_depth=5, random_state=42",
            "n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42",
            "hidden_layer_sizes=(128,128), activation='relu', solver='adam', max_iter=500, random_state=42",
        ],
    }
)

# -----------------------------
# SHAP
# -----------------------------
@st.cache_resource
def get_shap_objects():
    explainer = shap.TreeExplainer(best_tree_model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

explainer, shap_values = get_shap_objects()


# -----------------------------
# UI
# -----------------------------
st.title("Titanic Survival Prediction — MSIS 522 HW1")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.header("Executive Summary")

    st.write(
        """
This project analyzes the Titanic dataset to predict whether a passenger survived the disaster. 
The target variable is **Survived**, a binary outcome where 1 indicates survival and 0 indicates non-survival.

This problem is meaningful because the Titanic dataset captures how demographic and socioeconomic features such as gender, age, fare, and passenger class relate to survival outcomes. 
It serves as a classic binary classification problem for demonstrating a complete machine learning workflow on tabular data.

The workflow includes descriptive analytics, predictive modeling, model comparison, explainability, and an interactive prediction interface. 
Five models were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network (MLP).
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Best Model (F1)", best_model_name)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Key Finding")
    st.write(
        f"""
Based on the test-set evaluation in this app, **{best_model_name}** achieved the strongest F1 performance. 
Tree-based models generally perform well on tabular datasets because they can capture nonlinear relationships and interactions among passenger characteristics.
"""
    )

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.header("Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Survived", data=df, ax=ax)
        ax.set_title("Target Distribution: Survived")
        st.pyplot(fig)
        st.caption(
            "The dataset contains more passengers who did not survive than passengers who survived. "
            "This indicates a mild class imbalance in the target."
        )

    with col2:
        st.subheader("Survival by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Gender")
        st.pyplot(fig)
        st.caption(
            "Female passengers had much higher survival rates than male passengers. "
            "This suggests gender was one of the strongest predictors of survival."
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Passenger Class")
        st.pyplot(fig)
        st.caption(
            "Passengers in first class survived at much higher rates than those in third class. "
            "This highlights the role of socioeconomic status in survival outcomes."
        )

    with col4:
        st.subheader("Fare vs Survival")
        fig, ax = plt.subplots()
        sns.boxplot(x="Survived", y="Fare", data=df, ax=ax)
        ax.set_title("Fare vs Survival")
        st.pyplot(fig)
        st.caption(
            "Passengers who paid higher fares tended to survive more often. "
            "This pattern is consistent with the effect of passenger class."
        )

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        st.caption(
            "Most passengers were between about 20 and 40 years old. "
            "Age may influence vulnerability and priority during evacuation."
        )

    with col6:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_model.corr(), cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        st.caption(
            "The heatmap summarizes relationships among encoded modeling variables. "
            "It helps identify features that are likely to contribute to predictive performance."
        )

# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.header("Model Performance")

    st.write(
        """
This section summarizes the predictive performance of all five models trained in Part 2. 
It includes the model comparison table, F1 score comparison, ROC curves, and the final hyperparameter settings used for each model.
"""
    )

    st.subheader("Model Comparison Table")
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, use_container_width=True)

    st.subheader("F1 Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="F1", ax=ax)
    ax.set_title("Model Comparison by F1 Score")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (fpr, tpr, auc) in roc_dict.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for All Models")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Best Hyperparameters")
    st.write(
        """
The table below reports the selected hyperparameters used for each model. 
For Decision Tree, Random Forest, and XGBoost, the values come from GridSearchCV. 
For Logistic Regression and MLP, the table reports the final settings used in the notebook.
"""
    )
    st.dataframe(hyperparams_df, use_container_width=True)

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    st.header("Explainability & Interactive Prediction")

    st.subheader(f"SHAP Summary Plot ({best_tree_model_name})")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader(f"SHAP Feature Importance Bar Plot ({best_tree_model_name})")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("SHAP Waterfall Plot for One Test Prediction")
    sample_idx = st.slider("Choose a test sample", 0, len(X_test) - 1, 0)

    shap_explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist(),
    )

    fig = plt.figure()
    shap.plots.waterfall(shap_explanation, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.subheader("Interactive Prediction")

    selected_model_name = st.selectbox("Choose a model", list(models.keys()))
    selected_model = models[selected_model_name]

    c1, c2, c3 = st.columns(3)

    with c1:
        pclass = st.slider("Passenger Class", 1, 3, 3)
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.slider("Siblings / Spouses Aboard", 0, 8, 0)

    with c2:
        parch = st.slider("Parents / Children Aboard", 0, 6, 0)
        fare = st.slider("Fare", 0.0, 550.0, 32.0)
        sex = st.selectbox("Sex", ["male", "female"])

    with c3:
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    user_input = build_input_df(
        X.columns,
        pclass=pclass,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        sex=sex,
        embarked=embarked,
    )

    prediction = selected_model.predict(user_input)[0]

    if hasattr(selected_model, "predict_proba"):
        probability = float(selected_model.predict_proba(user_input)[0][1])
        st.write(f"**Predicted survival probability:** {probability:.4f}")
    else:
        probability = None

    if prediction == 1:
        st.success("Predicted Outcome: Survived")
    else:
        st.error("Predicted Outcome: Did Not Survive")

    st.write("**Encoded model input used for prediction:**")
    st.dataframe(user_input, use_container_width=True)
