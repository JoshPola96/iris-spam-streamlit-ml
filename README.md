# Machine Learning Web App Suite: Iris Classifier & Spam/Ham Detector

This project demonstrates end-to-end machine learning workflows for two classification tasks: predicting the species of an iris flower and classifying messages as either "spam" or "ham." Both models are deployed as interactive web apps using **Streamlit** and integrated into a single dashboard with tabs for easy navigation.

## 🧠 Project Highlights
- Developed two ML models:
  - **Iris Flower Classifier**: Predicts flower species based on sepal and petal dimensions.
  - **Spam/Ham Message Classifier**: Classifies whether a given message is spam or ham.
- Both models use different datasets and preprocessing techniques but follow a similar pipeline for training, evaluation, and deployment.

## ⚙️ Technical Workflow

### 1. Data Preprocessing & Feature Engineering
- **Iris Classifier**:
  - Handled missing values and scaled numerical features (petal and sepal sizes).
  - Applied feature engineering to create informative variables (if applicable).
  
- **Spam/Ham Classifier**:
  - Applied **text preprocessing**:
    - Tokenization, stopword removal, stemming/lemmatization.
    - **TF-IDF vectorization** to convert text data into numerical features.
  
### 2. Model Training & Optimization
- Tried multiple ML algorithms:
  - **Iris Classifier**: Logistic Regression, Support Vector Machine (SVM), Decision Trees, Random Forest, etc.
  - **Spam/Ham Classifier**: Naive Bayes, SVM, Decision Trees, Random Forest, etc.
- Optimized models using **GridSearchCV** for hyperparameter tuning and cross-validation.
  
### 3. Evaluation & Model Selection
- Used classification metrics like **accuracy**, **precision**, **recall**, and **F1-score** to evaluate model performance.
- Stored the best-performing model using `joblib` for further use and deployment.

### 4. Web App Deployment
- Built two interactive web apps using **Streamlit**:
  - **Iris Classifier**: Inputs include sliders for petal and sepal dimensions. Displays predicted flower species.
  - **Spam/Ham Classifier**: Inputs a text message and predicts whether the message is spam or ham.
- Integrated both apps under a single **multi-tab Streamlit interface** for a unified user experience.

## 🚀 Technologies Used
- **Python**:
  - Pandas
  - scikit-learn
  - NumPy
  - joblib (for model persistence)
  - NLTK/spaCy (for text preprocessing in Spam/Ham)
- **Streamlit** for front-end development and web app deployment

## ✅ Key Outcomes
- Developed and deployed two accurate ML models with clean, interactive UI.
- Demonstrated full machine learning lifecycle: data preprocessing, modeling, evaluation, and deployment.
- Gained hands-on experience with **Streamlit** for creating user-friendly web interfaces for machine learning models.

## 📜 How to Run the Project Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/JoshPola96/iris-spam-streamlit-ml
   cd iris-spam-streamlit-ml
   ```

2. Install the required dependencies:

3. Run the main app:
   ```bash
   streamlit run main.py
   ```

4. Open the provided local URL in your browser to interact with the app.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributions

Feel free to fork the repo and submit pull requests with improvements or additional features!
