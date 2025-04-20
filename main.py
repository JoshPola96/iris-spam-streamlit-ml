import streamlit as st

# Import the predictor module
from iris_streamlit import predict_species
from spamham_streamlit import predict_message

# Main application
def main():
    st.title("Iris Species Predictor")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Select a Page", ["Home", "PredictIris", "PredictSpamHam"])

    if page == "Home":
        home_page()
    elif page == "PredictIris":
        predict_iris_page()
    elif page == "PredictSpamHam":
        predict_spahham_page()
        

# Home page content
def home_page():
    st.header("Welcome to the Iris Species Predictor")
    st.write("This application allows you to predict the species of an Iris flower based on its measurements.")
    st.write("Use the navigation to go to the prediction page.")
    st.image("sample3_image.jpg")

# Prediction page content
def predict_iris_page():
    predict_species()
    
def predict_spahham_page():
    predict_message()

if __name__ == "__main__":
    main()
