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


# --------------------------------------------------
# Data + model loading
# --------------------------------------------------
@st.cache_data
def load_raw_data():
    return pd.read_csv("Titanic-Dataset.csv")


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df_model = df.copy()

    # same preprocessing as notebook
    df_model["Age"] = df_model["Age"].fillna(df_model["Age"].median())
    df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])

    df_model.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
        inplace=True,
    )

    df_model = pd.get_dummies(df_model, drop_first=True)

    # make sure bool columns become int for model compatibility
    for col in df_model.columns:
        if df_model[col].dtype == bool:
            df_model[col] = df_model[col].astype(int)

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
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_dict[model_name] = (fpr, tpr, auc)

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

    # numeric
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

    # encoded categorical
    if "Sex_male" in input_dict:
        input_dict["Sex_male"] = 1 if sex == "male" else 0

    # drop_first=True => C is baseline
    if "Embarked_Q" in input_dict:
        input_dict["Embarked_Q"] = 1 if embarked == "Q" else 0
    if "Embarked_S" in input_dict:
        input_dict["Embarked_S"] = 1 if embarked == "S" else 0

    user_df = pd.DataFrame([input_dict])

    # enforce column order
    user_df = user_df[X_columns]

    return user_df


# --------------------------------------------------
# Load runtime objects
# --------------------------------------------------
df = load_raw_data()
df_model, X, y = preprocess_data(df)
models = load_models()
X_train, X_test, y_train, y_test = split_data(X, y)
results_df, roc_dict = evaluate_models(X_test, y_test)

best_model_name = results_df.iloc[0]["Model"]  # should be Logistic Regression from notebook
best_tree_model_name = "XGBoost"
best_tree_model = models[best_tree_model_name]

# actual model-performance table from notebook results
actual_results_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "Neural Network (MLP)",
        ],
        "Accuracy": [0.809701, 0.772388, 0.817164, 0.802239, 0.787313],
        "Precision": [0.794118, 0.784091, 0.860465, 0.784314, 0.821429],
        "Recall": [0.729730, 0.621622, 0.666667, 0.720721, 0.621622],
        "F1": [0.760563, 0.693467, 0.751269, 0.751174, 0.707692],
        "AUC": [0.880588, 0.815918, 0.871349, 0.856602, 0.862570],
    }
).sort_values(by="F1", ascending=False).reset_index(drop=True)

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
            "max_depth=7, min_samples_leaf=10",
            "n_estimators=100, max_depth=5, random_state=42",
            "n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'",
            "hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=500, random_state=42",
        ],
    }
)

# SHAP objects for best tree-based model
@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(best_tree_model)

explainer = get_shap_explainer()
shap_values_test = explainer.shap_values(X_test)


# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("Titanic Survival Prediction — MSIS 522 HW1")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

# --------------------------------------------------
# TAB 1 — Executive Summary
# --------------------------------------------------
with tab1:
    st.header("Executive Summary")

    st.write(
        """
This project analyzes the Titanic dataset to predict whether a passenger survived the disaster. 
The dataset contains 891 passenger records and 12 variables, including demographic information, travel class, ticket fare, and embarkation details. 
The target variable is **Survived**, where 1 indicates survival and 0 indicates non-survival.
"""
    )

    st.write(
        """
This problem matters because it shows how machine learning can be used to identify the passenger characteristics most associated with survival outcomes. 
From a business and decision-making perspective, it is a compact example of how predictive analytics can uncover high-impact drivers, compare modeling trade-offs, and translate patterns into actionable insights for non-technical stakeholders.
"""
    )

    st.write(
        """
The workflow included exploratory data analysis, preprocessing, five classification models, cross-validated hyperparameter tuning for tree-based methods, ROC analysis, and SHAP explainability. 
Among all models, **Logistic Regression performed best by F1 score (0.7606)**, while it also achieved the strongest AUC (0.8806). 
This suggests that for this Titanic feature set, a relatively simple linear decision boundary generalized slightly better than the more flexible nonlinear models.
"""
    )

    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Best Model", "Logistic Regression")

    st.subheader("Sample Records")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Key Findings")
    st.markdown(
        """
- The strongest survival patterns in the Titanic dataset were associated with gender, passenger class, and fare. Female passengers and first-class passengers were much more likely to survive, while lower-class status and being male were associated with lower survival probability. 
- Across five models, Logistic Regression achieved the strongest F1 score and AUC on the test set, slightly outperforming the more complex tree-based and neural models. This suggests that the Titanic survival signal in this feature set is relatively structured and can be captured effectively even by a simpler baseline model.
- SHAP analysis from the best-performing tree-based model showed that **Sex_male, Pclass, Fare, and Age** were the most influential features in predicting survival.
"""
    )

# --------------------------------------------------
# TAB 2 — Descriptive Analytics
# --------------------------------------------------
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
            "This indicates a mild class imbalance, but not one so extreme that it prevents standard binary classification modeling."
        )

    with col2:
        st.subheader("Survival by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Gender")
        st.pyplot(fig)
        st.caption(
            "Female passengers survived at much higher rates than male passengers. "
            "This large separation suggests that gender is likely to be one of the most predictive variables in the model."
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Passenger Class")
        st.pyplot(fig)
        st.caption(
            "First-class passengers had substantially better survival outcomes than third-class passengers. "
            "This pattern suggests that social and economic status likely affected access to safety and evacuation."
        )

    with col4:
        st.subheader("Fare vs Survival")
        fig, ax = plt.subplots()
        sns.boxplot(x="Survived", y="Fare", data=df, ax=ax)
        ax.set_title("Fare vs Survival")
        st.pyplot(fig)
        st.caption(
            "Passengers who survived generally paid higher fares on average. "
            "This reinforces the class-based survival pattern and suggests that fare captures additional socioeconomic information."
        )

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        st.caption(
            "Most passengers were concentrated between roughly ages 20 and 40. "
            "Age may still matter in prediction, but its effect appears less visually dramatic than gender or class."
        )

    with col6:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        st.caption(
            "The heatmap shows that survival is negatively correlated with `Sex_male` and `Pclass`, and positively correlated with `Fare`. "
            "These relationships are consistent with the visual patterns and help explain why those variables are important in modeling."
        )

# --------------------------------------------------
# TAB 3 — Model Performance
# --------------------------------------------------
with tab3:
    st.header("Model Performance")

    st.write(
        """
This section surfaces the main outputs from Part 2 of the assignment: the model comparison table, ROC curves, and final hyperparameter settings. 
Together, these results allow a reader to compare predictive performance, generalization trade-offs, and modeling complexity without opening the notebook.
"""
    )

    st.subheader("Model Comparison Table")
    display_df = actual_results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, use_container_width=True)

    st.subheader("F1 Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=actual_results_df, x="Model", y="F1", ax=ax)
    ax.set_title("Model Comparison by F1 Score")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("ROC Curves for All Models")
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (fpr, tpr, auc) in roc_dict.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("MLP Training Loss Curve")
    mlp_model = models["Neural Network (MLP)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp_model.loss_curve_)
    ax.set_title("MLP Training Loss Curve")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

    st.subheader("Best Hyperparameters")
    st.write(
        """
The table below reports the selected hyperparameters used for each model. 
For Decision Tree, Random Forest, and XGBoost, the values come directly from GridSearchCV in the notebook. 
For Logistic Regression and MLP, the table reports the final settings used for training.
"""
    )
    st.dataframe(hyperparams_df, use_container_width=True)

    st.subheader("Performance Interpretation")
    st.write(
        """
Logistic Regression achieved the highest F1 score, which was somewhat surprising given that tree ensembles often perform best on tabular data. 
However, the margins were small: Random Forest and XGBoost were both highly competitive, while Decision Tree and MLP lagged slightly. 
This outcome suggests a trade-off between interpretability and complexity: the linear baseline was both simple and strong, while the ensembles provided richer nonlinear modeling at the cost of transparency.
"""
    )

# --------------------------------------------------
# TAB 4 — Explainability & Interactive Prediction
# --------------------------------------------------
with tab4:
    st.header("Explainability & Interactive Prediction")

    st.subheader(f"SHAP Summary Plot ({best_tree_model_name})")
    fig = plt.figure()
    shap.summary_plot(shap_values_test, X_test, show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader(f"SHAP Feature Importance Bar Plot ({best_tree_model_name})")
    fig = plt.figure()
    shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
    st.pyplot(fig, clear_figure=True)

    st.write(
        """
The SHAP plots show that **Sex_male**, **Pclass**, **Fare**, and **Age** have the strongest impact on predictions. 
Being male and belonging to a lower passenger class generally push predictions toward non-survival, while higher fares and more favorable class positions push predictions toward survival.
"""
    )

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
    probability = float(selected_model.predict_proba(user_input)[0][1])

    st.write(f"**Predicted survival probability:** {probability:.4f}")

    if prediction == 1:
        st.success("Predicted Outcome: Survived")
    else:
        st.error("Predicted Outcome: Did Not Survive")

    st.write("**Encoded model input used for prediction:**")
    st.dataframe(user_input, use_container_width=True)

    # custom-input SHAP waterfall using best tree-based model
    st.subheader(f"SHAP Waterfall Plot for Custom Input ({best_tree_model_name})")
    user_shap_values = explainer.shap_values(user_input)

    user_explanation = shap.Explanation(
        values=user_shap_values[0],
        base_values=explainer.expected_value,
        data=user_input.iloc[0],
        feature_names=user_input.columns.tolist(),
    )

    fig = plt.figure()
    shap.plots.waterfall(user_explanation, show=False)
    st.pyplot(fig, clear_figure=True)

    st.write(
        """
This waterfall plot explains the prediction for the custom passenger profile you entered above. 
It shows which features moved the prediction upward toward survival probability and which features pushed it downward toward non-survival.
"""
    )
