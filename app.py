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
    confusion_matrix,
)
from sklearn.tree import plot_tree

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")


# ======================================================
# Helpers
# ======================================================
@st.cache_data
def load_raw_data():
    return pd.read_csv("Titanic-Dataset.csv")


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df_model = df.copy()

    # Same preprocessing as notebook
    df_model["Age"] = df_model["Age"].fillna(df_model["Age"].median())
    df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])

    df_model.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
        inplace=True,
    )

    df_model = pd.get_dummies(df_model, drop_first=True)

    # Convert bool columns to int to avoid model compatibility issues
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
def get_eval_outputs(X_test, y_test):
    models = load_models()

    roc_dict = {}
    confusion_dict = {}
    live_rows = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_dict[model_name] = (fpr, tpr, auc)
        confusion_dict[model_name] = confusion_matrix(y_test, y_pred)

        live_rows.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC": auc,
            }
        )

    live_results_df = pd.DataFrame(live_rows)
    return roc_dict, confusion_dict, live_results_df


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

    # numeric features
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

    # encoded categorical variables from pd.get_dummies(drop_first=True)
    if "Sex_male" in input_dict:
        input_dict["Sex_male"] = 1 if sex == "male" else 0

    # Embarked: C is baseline when drop_first=True
    if "Embarked_Q" in input_dict:
        input_dict["Embarked_Q"] = 1 if embarked == "Q" else 0
    if "Embarked_S" in input_dict:
        input_dict["Embarked_S"] = 1 if embarked == "S" else 0

    user_df = pd.DataFrame([input_dict])
    user_df = user_df[X_columns]  # enforce exact column order
    return user_df


def _standardize_shap_values(shap_values):
    """
    SHAP may return:
    - ndarray of shape (n_samples, n_features)
    - list of arrays for classification
    This helper standardizes to ndarray (n_samples, n_features).
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    return shap_values


def _scalar_expected_value(expected_value):
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).flatten()
        return float(expected_value[-1])
    return float(expected_value)


# ======================================================
# Load runtime objects
# ======================================================
df = load_raw_data()
df_model, X, y = preprocess_data(df)
models = load_models()
X_train, X_test, y_train, y_test = split_data(X, y)
roc_dict, confusion_dict, live_results_df = get_eval_outputs(X_test, y_test)

# Use actual notebook result table for exact reporting
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

best_model_name = "Logistic Regression"
best_tree_model_name = "Random Forest"  # best-performing tree-based model by F1
best_tree_model = models[best_tree_model_name]

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

@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(best_tree_model)

explainer = get_shap_explainer()
shap_values_test = _standardize_shap_values(explainer.shap_values(X_test))
expected_value_scalar = _scalar_expected_value(explainer.expected_value)


# ======================================================
# UI
# ======================================================
st.title("Titanic Survival Prediction — MSIS 522 HW1")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

# ======================================================
# Tab 1 — Executive Summary
# ======================================================
with tab1:
    st.header("Executive Summary")

    st.write(
        """
This project analyzes the Titanic passenger dataset to predict whether a passenger survived the disaster. 
The dataset contains 891 observations and 12 original variables describing demographic characteristics, travel class, fare, embarkation port, and family relationships. 
The prediction target is **Survived**, a binary variable where 1 indicates survival and 0 indicates non-survival.
"""
    )

    st.write(
        """
This problem is meaningful because it provides a compact but realistic example of how machine learning can uncover which passenger characteristics were most strongly associated with survival outcomes. 
From a decision-making perspective, the task demonstrates how structured data can be used not only for prediction, but also for explaining which factors most influence individual outcomes.
"""
    )

    st.write(
        """
The workflow included descriptive analytics, preprocessing, multiple classification models, cross-validated hyperparameter tuning, ROC analysis, and SHAP explainability. 
Five models were trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network (MLP). 
Among all models, **Logistic Regression achieved the best overall performance**, with the highest **F1 score (0.7606)** and **AUC (0.8806)** on the held-out test set.
"""
    )

    st.write(
        """
The results suggest that the survival signal in this Titanic feature set is relatively structured and can be captured effectively even by a simple linear baseline. 
At the same time, SHAP analysis using the best-performing tree-based model showed that **gender, passenger class, fare, and age** were the most influential drivers of survival probability. 
Overall, this project demonstrates both predictive modeling performance and interpretable insights for stakeholders who want to understand not just *what* the model predicts, but *why*.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", 891)
    c2.metric("Columns", 12)
    c3.metric("Target", "Survived")
    c4.metric("Best Model", "Logistic Regression")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Feature Overview")
    st.write(
        """
The original dataset contains both numerical and categorical variables. 
Numerical variables include PassengerId, Survived, Pclass, Age, SibSp, Parch, and Fare, while categorical variables include Name, Sex, Ticket, Cabin, and Embarked.
"""
    )

# ======================================================
# Tab 2 — Descriptive Analytics
# ======================================================
with tab2:
    st.header("Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Survived", data=df, ax=ax)
        ax.set_title("Target Distribution: Survived")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The dataset contains more passengers who did not survive than passengers who survived. "
            "The class imbalance is moderate rather than extreme, so standard binary classification models remain appropriate while still making F1 and AUC useful evaluation metrics."
        )

    with col2:
        st.subheader("Survival by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Gender")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Female passengers survived at much higher rates than male passengers. "
            "This large separation suggests that gender is one of the strongest predictors of survival in the dataset."
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
        ax.set_title("Survival by Passenger Class")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Passengers in first class had much higher survival rates than passengers in third class. "
            "This indicates that socioeconomic status likely influenced access to safety resources and evacuation opportunities."
        )

    with col4:
        st.subheader("Fare vs Survival")
        fig, ax = plt.subplots()
        sns.boxplot(x="Survived", y="Fare", data=df, ax=ax)
        ax.set_title("Fare vs Survival")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Passengers who survived generally paid higher fares on average. "
            "Because fare is closely related to passenger class, it appears to act as an additional proxy for status and travel conditions."
        )

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Most passengers were concentrated between roughly ages 20 and 40, with a right-skewed tail into older ages. "
            "Age may influence survival, but its effect appears more subtle than the stronger gender and class differences visible in the other plots."
        )

    with col6:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The heatmap shows that survival is negatively correlated with `Sex_male` and `Pclass`, and positively correlated with `Fare`. "
            "These correlations are consistent with the visual patterns above and suggest that gender, class, and fare should play an important role in predictive modeling."
        )

# ======================================================
# Tab 3 — Model Performance
# ======================================================
with tab3:
    st.header("Model Performance")

    st.write(
        """
This section surfaces the main outputs from Part 2 of the assignment so that a reader can evaluate all models without opening the notebook. 
It includes the model comparison table, F1 comparison chart, ROC curves, selected hyperparameters, and additional diagnostics such as the MLP loss curve and the best decision tree visualization.
"""
    )

    st.subheader("Data Preparation Summary")
    st.write(
        """
The modeling pipeline used `Survived` as the target variable and all remaining processed variables as features. 
Missing values in **Age** were filled with the median, missing values in **Embarked** were filled with the mode, and identifier-like columns such as PassengerId, Name, Ticket, and Cabin were dropped. 
Categorical variables were then one-hot encoded, and the data was split into a 70/30 train-test split using `random_state=42`.
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
    plt.close(fig)

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
    plt.close(fig)

    st.subheader("Best Hyperparameters")
    st.write(
        """
The table below reports the selected hyperparameters used for each model. 
For Decision Tree, Random Forest, and XGBoost, the values come from GridSearchCV. 
For Logistic Regression and MLP, the table reports the final settings used in the notebook.
"""
    )
    st.dataframe(hyperparams_df, use_container_width=True)

    st.subheader("MLP Training Loss Curve")
    mlp_model = models["Neural Network (MLP)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp_model.loss_curve_)
    ax.set_title("MLP Training Loss Curve")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Show Best Decision Tree Visualization"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            models["Decision Tree"],
            feature_names=X.columns,
            class_names=["Not Survived", "Survived"],
            filled=True,
            fontsize=8,
            ax=ax,
        )
        ax.set_title("Best Decision Tree")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Show Confusion Matrices for All Models"):
        cols = st.columns(2)
        model_names = list(models.keys())
        for i, model_name in enumerate(model_names):
            cm = confusion_dict[model_name]
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(model_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            cols[i % 2].pyplot(fig)
            plt.close(fig)

    st.subheader("Performance Interpretation")
    st.write(
        """
Logistic Regression achieved the highest F1 score and the highest AUC, which was somewhat surprising given that ensemble tree models often dominate tabular tasks. 
However, the margins were small: Random Forest and XGBoost were both highly competitive, and Random Forest delivered the strongest precision among all models. 
This highlights a useful trade-off: the simpler baseline model was the overall winner, while the more complex tree-based models remained attractive when interpretability through feature interactions or precision-focused behavior is important.
"""
    )

# ======================================================
# Tab 4 — Explainability & Interactive Prediction
# ======================================================
with tab4:
    st.header("Explainability & Interactive Prediction")

    st.write(
        f"""
SHAP analysis is performed using **{best_tree_model_name}**, the best-performing tree-based model in the notebook. 
This aligns with the assignment requirement to explain predictions using the strongest tree-based model while still allowing users to generate predictions from any of the five saved models.
"""
    )

    st.subheader(f"SHAP Summary Plot ({best_tree_model_name})")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_test, X_test, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.subheader(f"SHAP Feature Importance Bar Plot ({best_tree_model_name})")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.write(
        """
The SHAP plots show that **Sex_male, Pclass, Fare, and Age** have the strongest impact on survival predictions. 
Being male and belonging to a lower passenger class generally push predictions toward non-survival, while higher fares and more favorable class positions push predictions toward survival.
"""
    )

    st.write(
        """
For decision-makers, these insights are useful because they show that the model is learning meaningful patterns rather than relying on random correlations. 
This improves trust in the model and makes it easier to explain predicted outcomes to non-technical stakeholders.
"""
    )

    st.markdown("---")
    st.subheader("Interactive Prediction")

    selected_model_name = st.selectbox("Choose a model for prediction", list(models.keys()))
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

    st.subheader(f"SHAP Waterfall Plot for Custom Input ({best_tree_model_name})")
    user_shap_values = _standardize_shap_values(explainer.shap_values(user_input))

    user_explanation = shap.Explanation(
        values=user_shap_values[0],
        base_values=expected_value_scalar,
        data=user_input.iloc[0],
        feature_names=user_input.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(user_explanation, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.write(
        """
This waterfall plot explains the prediction for the custom passenger profile entered above. 
It shows which features increased the predicted survival probability and which features pushed the prediction downward toward non-survival.
"""
    )
