import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# info abt pg
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

# importin data
df = pd.read_csv("Housing.csv")  # Load dataset for EDA
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# side bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["EDA", "Prediction"])

# EDA
if page == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    col1, col2 = st.columns(2)

    # Price Distribution
    with col1:
        st.write("### Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["price"]/1e6, bins=20, edgecolor="black")
        ax.set_xlabel("Price (millions)")
        ax.set_ylabel("No. of Houses")
        st.pyplot(fig)

    # Area Distribution
    with col2:
        st.write("### Area Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["area"], bins=20, color="orange", edgecolor="black")
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("No. of Houses")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Price vs Bedrooms
    with col3:
        st.write("### Price vs Bedrooms")
        fig, ax = plt.subplots()
        sns.boxplot(x="bedrooms", y="price", data=df, ax=ax)
        st.pyplot(fig)

    # Area vs Price
    with col4:
        st.write("### Area vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x="area", y="price", data=df, ax=ax)
        st.pyplot(fig)

# prediction
elif page == "Prediction":
    st.subheader("Enter House Features")

    # Numeric inputs in two columns
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, step=100, value=7420)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    with col2:
        stories = st.number_input("Stories", min_value=1, max_value=5, value=3)
        parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=2)

    # Yes/No categorical features
    def encode_yes_no(value): return 1 if value == "yes" else 0

    col3, col4, col5 = st.columns(3)

    with col3:
        mainroad_yes = encode_yes_no(st.radio("Main Road?", ["yes", "no"], index=0))
        guestroom_yes = encode_yes_no(st.radio("Guest Room?", ["yes", "no"], index=1))

    with col4:
        basement_yes = encode_yes_no(st.radio("Basement?", ["yes", "no"], index=1))
        hotwaterheating_yes = encode_yes_no(st.radio("Hot Water Heating?", ["yes", "no"], index=1))

    with col5:
        airconditioning_yes = encode_yes_no(st.radio("Air Conditioning?", ["yes", "no"], index=0))
        prefarea_yes = encode_yes_no(st.radio("Preferred Area?", ["yes", "no"], index=0))

    # Furnishing status
    furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
    furnishing_semi = 1 if furnishing == "semi-furnished" else 0
    furnishing_un = 1 if furnishing == "unfurnished" else 0

    # Create feature array
    features = np.array([[area, bedrooms, bathrooms, stories, parking,
                          mainroad_yes, guestroom_yes, basement_yes,
                          hotwaterheating_yes, airconditioning_yes, prefarea_yes,
                          furnishing_semi, furnishing_un]])

    # Prediction Button
    if st.button("Predict Price"):
        prediction = model.predict(features)
        st.success(f"🏡 Predicted House Price: **${prediction[0]:,.2f}**")


