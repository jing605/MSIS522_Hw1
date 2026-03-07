import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Titanic Survival Analysis")

df = pd.read_csv("Titanic-Dataset.csv")

st.header("Dataset Preview")
st.dataframe(df.head())

st.header("Survival Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Survived", data=df, ax=ax)
st.pyplot(fig)

st.header("Survival by Gender")

fig, ax = plt.subplots()
sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
st.pyplot(fig)

st.header("Survival by Passenger Class")

fig, ax = plt.subplots()
sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
st.pyplot(fig)
