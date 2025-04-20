import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to check and download NLTK resources
def download_nltk_resources():
    resources = ['wordnet', 'stopwords', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(resource)
            print(f"Resource '{resource}' is already downloaded.")
        except LookupError:
            print(f"Downloading resource '{resource}'...")
            nltk.download(resource)

# Download required NLTK resources
download_nltk_resources()

# Load dataset
df = pd.read_csv('D:\\personal\\Code\\Irohub_DS\\7 - Streamlit\\IRIS\\spamham_data.csv', sep='\t', names=['class', 'text'], header=None)

# Encode labels
df['class'] = df['class'].map({'ham': 0, 'spam': 1})

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

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'spamham_tfidf_vectorizer.joblib')

# Initialize models
models = {
    "LogisticRegressionModel": LogisticRegression(),
    "SVCModel": SVC(),
    "MultinomialNBModel": MultinomialNB(),
    "RandomForestClassifierModel": RandomForestClassifier(),
    "XGBClassifierModel": XGBClassifier(),
    "KNeighborsClassifierModel": KNeighborsClassifier(),
    "MLPClassifierModel": MLPClassifier()
}

# Parameter grid for hyperparameter tuning
param_grid = {
    'LogisticRegressionModel': [{'C': val} for val in [0.01, 0.1, 1, 10, 100]],
    'SVCModel': [{'C': val} for val in [0.01, 0.1, 1, 10, 100]],
    'MultinomialNBModel': [{'alpha': val} for val in [0.01, 0.1, 1, 10]],
    'RandomForestClassifierModel': [{'n_estimators': n, 'max_depth': d} for n in [50, 100, 200] for d in [None, 10, 20, 30]],
    'XGBClassifierModel': [{'n_estimators': n, 'learning_rate': lr, 'max_depth': d} for n in [50, 100, 200] for lr in [0.01, 0.1, 0.2] for d in [3, 5, 7]],
    'KNeighborsClassifierModel': [{'n_neighbors': n} for n in [3, 5, 7, 9]],
    'MLPClassifierModel': [{'hidden_layer_sizes': size, 'learning_rate_init': lr} for size in [(50,), (100,), (50, 50)] for lr in [0.001, 0.01, 0.1]],
}

# Evaluate function
def evaluate_model(model, params, X_train, y_train, X_test, y_test):
    results = []
    for param in params:
        model.set_params(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Print evaluation metrics
        print(f"Model: {model.__class__.__name__}, Params: {param}, Accuracy: {accuracy:.4f}, "
              f"Precision: {report['weighted avg']['precision']:.4f}, "
              f"Recall: {report['weighted avg']['recall']:.4f}, "
              f"F1-score: {report['weighted avg']['f1-score']:.4f}")

        # Store results
        results.append({
            'params': param,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        })

    return results

# Store results
results = []

# Model evaluation
for name, model in models.items():
    for param in param_grid[name]:
        metrics = evaluate_model(model, [param], X_train_tfidf, y_train, X_test_tfidf, y_test)
        for metric in metrics:
            results.append({
                'Model': name,
                'Params': metric['params'],
                'Accuracy': metric['accuracy'],
                'Precision': metric['precision'],
                'Recall': metric['recall'],
                'F1-score': metric['f1-score']
            })

# Create DataFrame for results
results_df = pd.DataFrame(results)

# Group and find the maximum for desired metrics
top_results = results_df.groupby('Model').agg({
    'F1-score': 'max',
    'Accuracy': 'max',
    'Precision': 'max',
    'Recall': 'max'
}).reset_index()

# Merge back to get the complete rows from the original DataFrame
top_results = pd.merge(top_results, results_df, on=['Model', 'F1-score', 'Accuracy', 'Precision', 'Recall'])
top_results = top_results.sort_values(by=['F1-score', 'Accuracy', 'Precision', 'Recall'], ascending=False)
top_results = top_results.drop_duplicates(subset='Model')

# Display the final results dataset
print("\nFinal Results Dataset:")
print(top_results)

# Sample texts for prediction
sample_texts = [
    "Congratulations! You've won a free ticket to Bahamas.",  # Spam
    "Dear user, your account will be suspended unless you verify your details.",  # Spam
    "Hey, are we still on for lunch tomorrow?",  # Ham
    "You have received a $1000 Walmart gift card. Click here to claim now!",  # Spam
    "Just checking in to see how your project is going.",  # Ham
    "Limited time offer! Get a discount on your next purchase.",  # Spam
    "Hi! It was great to see you at the event last week.",  # Ham
    "Urgent: Your invoice is overdue. Please pay now to avoid penalties.",  # Spam
    "Can you send me the report by end of the day?",  # Ham
    "Win a new car! Enter your email to participate.",  # Spam
]

# Train the best model
best_model_name = top_results.iloc[0]['Model']
best_params = top_results.iloc[0]['Params']
best_model_instance = models[best_model_name].__class__()  # Create a new instance
best_model_instance.set_params(**best_params)
best_model_instance.fit(X_train_tfidf, y_train)

# Save the best model
joblib.dump(best_model_instance, 'spamham_best_model.joblib')

# Display the selected best model and its parameters
print("\nSelected Best Model:")
print(f"Model: {best_model_name}, Parameters: {best_params}")

# Prediction function
def predict_custom_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = best_model_instance.predict(text_tfidf)
    return 'spam' if prediction[0] == 1 else 'ham'

# Test predictions
for text in sample_texts:
    print(f"Custom Text: '{text}' -> Prediction: {predict_custom_text(text)}")
