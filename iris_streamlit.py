import pandas as pd
import streamlit as st
import joblib

# Constants for file paths
MODEL_PATH = 'iris_best_model.joblib'
SCALER_PATH = 'iris_scaler.joblib'
LABEL_ENCODER_PATH = 'iris_label_encoder.joblib'

# Load the model, scaler, and label encoder
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()  

# Prediction page functionality
def predict_species():
    model, scaler, le = load_model()

    # Input sliders for user parameters
    st.header("Input Parameters")

    # Sliders for user inputs
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    # Prepare input data for prediction as a DataFrame
    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })

    # Button to make prediction
    if st.button("Predict Species"):
        try:
            # Scale the input data
            scaled_data = scaler.transform(input_data)  # Scale using the loaded scaler
            predictions = model.predict(scaled_data)  # Corrected method name
            predicted_species = le.inverse_transform(predictions)

            # Display the prediction
            st.success(f"Predicted Species: **{predicted_species[0]}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
