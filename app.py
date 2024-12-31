import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Title of the app
st.title("✨ Predictive Maintenance: RUL Prediction ✨")

# Description
st.markdown("""
Welcome to the Predictive Maintenance App. This app helps you predict the Remaining Useful Life (RUL) of your machines. 
Specify the number of features and provide the corresponding readings to get instant insights!
""")

# Sidebar
st.sidebar.header("Model Settings")

# Allow the user to select the number of features
num_features = st.sidebar.number_input("Number of Features", min_value=1, max_value=50, value=19, step=1)
st.sidebar.markdown(f"**Selected Number of Features: {num_features}**")

# Input fields for the feature readings
st.sidebar.header("Input Features")
input_data = []
for i in range(num_features):
    value = st.sidebar.number_input(f"Feature {i + 1}", min_value=0.0, max_value=1.0, step=0.01)
    input_data.append(value)

# Ensure input_data is correctly formatted as a numpy array of floats
input_data = np.array([float(i) for i in input_data]).reshape(1, -1)

# Load the pre-trained model
model = joblib.load('predictive_model.pkl')

# Predict RUL
if st.button("Predict RUL"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🛠 Predicted Remaining Useful Life (RUL): {prediction[0]:.2f} cycles")
    except ValueError as e:
        st.error(f"Error: {e}")

# Data Visualization (Example: Feature Importance)
st.subheader("Feature Importance Visualization")
st.markdown("Understand which features have the most impact on the prediction.")

# Dummy data for feature importance (replace with actual model feature importance if available)
feature_importance = np.random.rand(num_features)
feature_names = [f"Feature {i+1}" for i in range(num_features)]

# Create a DataFrame for visualization
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Bar chart for feature importance
st.bar_chart(feature_df.set_index('Feature'))

# Footer
st.markdown("---")
st.markdown("© 2024 Predictive Maintenance Inc.")
