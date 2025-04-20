import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK setup (if not already downloaded)
def download_nltk_resources():
    resources = ['wordnet', 'stopwords', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(resource)
            print(f"Resource '{resource}' is already downloaded.")
        except LookupError:
            print(f"Downloading resource '{resource}'...")
            nltk.download(resource)

# Constants for file paths
MODEL_PATH = 'spamham_best_model.joblib'
SCALER_PATH = 'spamham_tfidf_vectorizer.joblib'

# Load the model and vectorizer
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(SCALER_PATH)
        return model, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()  # Stop the execution if the model files cannot be loaded

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

# Prediction page functionality
def predict_message():
    model, tfidf_vectorizer = load_model()

    # Input text area for user input
    st.header("Spam or Ham Classifier")
    user_input = st.text_area("Enter your message:")

    # Button to make prediction
    if st.button("Predict"):
        if user_input:
            try:
                # Preprocess the input text
                processed_text = preprocess_text(user_input)

                # Transform input text using TF-IDF vectorizer
                text_tfidf = tfidf_vectorizer.transform([processed_text])
                prediction = model.predict(text_tfidf)

                # Display the prediction
                prediction_label = 'Spam' if prediction[0] == 1 else 'Ham'
                st.success(f"Prediction: **{prediction_label}**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a message to classify.")

# Run the application
if __name__ == "__main__":
    st.set_page_config(page_title="Spam or Ham Classifier", layout="wide")
    predict_text()
