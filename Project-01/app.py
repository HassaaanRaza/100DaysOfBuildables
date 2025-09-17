import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib


def main():

    # MAIN PAGE SETUP:
    st.set_page_config(page_title="Income Prediction", page_icon="💼")
    st.title("💼 INCOME PREDICTION")

    df = get_dataset()
    model = load_model()

    # taking input numeric data
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    education_num = st.number_input("Education Number (1-16)", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)

    # taking input categorical data
    sex = st.selectbox("Sex", ["Male", "Female"])
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", 
        "Amer-Indian-Eskimo", "Other"
    ])
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse"
    ])
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", 
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
        "Transport-moving", "Priv-house-serv", "Protective-serv", 
        "Armed-Forces"
    ])
    relationship = st.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ])
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada",
        "India", "England", "China", "Cuba", "Jamaica", "South", "Japan",
        "Other"
    ])

    if st.button("Predict Income"):

        input_df = pd.DataFrame([{
            "age": age,
        "hours.per.week": hours_per_week,
        "education.num": education_num,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "sex": sex,
        "race": race,
        "workclass": workclass,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "native.country": native_country,
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Income Category: **{prediction}**")

@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.joblib")

@st.cache_data
def get_dataset():
    return pd.read_csv("clean_adult.csv")







if __name__ == "__main__":
    main()
