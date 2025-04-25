import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="NYC Taxi Fare Prediction", layout="wide")

st.title("🚕 NYC Taxi Fare & Trip Duration Analysis")

# Load model and preprocessor
model = joblib.load("taxi_model.pkl")
preprocessor = joblib.load("taxi_preprocessor.pkl")

# Sidebar - user input
st.sidebar.header("Enter Trip Details")
trip_distance = st.sidebar.slider("Trip Distance (miles)", 0.1, 30.0, 2.5)
hour_of_day = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)
pickup_zone = st.sidebar.selectbox("Pickup Zone", [1, 2, 3, 4, 5])  # Customize as needed

# Predict
input_df = pd.DataFrame([{
    'trip_distance': trip_distance,
    'hour_of_day': hour_of_day,
    'pickup_zone': pickup_zone
}])
transformed = preprocessor.transform(input_df)
prediction = model.predict(transformed)[0]

st.subheader("📊 Prediction")
st.write(f"💰 **Estimated Fare or Duration**: ${prediction:.2f}")

# Load sample data for visualizations
st.subheader("🔍 Research Questions Visualizations")

# Load sample cleaned dataset
data = pd.read_csv("cleaned_taxi_data.csv")  # Make sure this file is in the same folder

# Question 1: What factors influence trip duration?
st.markdown("**1. Key Factors Influencing Trip Duration**")
fig1 = plt.figure(figsize=(10, 4))
sns.boxplot(x="hour_of_day", y="trip_duration", data=data)
plt.title("Trip Duration by Hour of Day")
st.pyplot(fig1)

# Question 2: Fare prices & tipping behavior
st.markdown("**2. Fare Prices and Tipping Behavior by Distance**")
fig2 = plt.figure(figsize=(10, 4))
sns.scatterplot(x="trip_distance", y="fare_amount", hue="tip_amount", data=data)
plt.title("Fare vs Distance Colored by Tip")
st.pyplot(fig2)

# Question 3: Model accuracy
st.markdown("**3. Random Forest Performance (MAE, RMSE, R²)**")
metrics = pd.read_csv("model_metrics.csv")  # CSV with performance results
st.dataframe(metrics)

st.success("✅ App loaded successfully.")
