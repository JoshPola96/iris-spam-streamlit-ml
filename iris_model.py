import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree, discriminant_analysis
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants for file paths
DATASET_PATH = "D:/personal/Code/Irohub_DS/7 - Streamlit/IRIS/iris_data.csv"
MODEL_OUTPUT_PATH = 'iris_best_model.joblib'
SCALER_OUTPUT_PATH = 'iris_scaler.joblib'
LE_OUTPUT_PATH = 'iris_label_encoder.joblib'

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a specified path."""
    try:
        df = pd.read_csv(path, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the dataset."""
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    
    X = df.drop(columns=['Species'])
    Y = df['Species']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    if np.any(np.isnan(X_scaled)) or np.any(np.isnan(Y)):
        logging.warning("Data contains NaN values.")
    
    return X_scaled, Y, le, scaler

def train_models(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """Train various models and return evaluation metrics."""
    models = {
        'AdaBoostClassifier': ensemble.AdaBoostClassifier(algorithm='SAMME'),
        'BaggingClassifier': ensemble.BaggingClassifier(),
        'ExtraTreesClassifier': ensemble.ExtraTreesClassifier(),
        'GradientBoostingClassifier': ensemble.GradientBoostingClassifier(),
        'RandomForestClassifier': ensemble.RandomForestClassifier(),
        'GaussianProcessClassifier': gaussian_process.GaussianProcessClassifier(),
        'LogisticRegressionCV': linear_model.LogisticRegressionCV(),
        'PassiveAggressiveClassifier': linear_model.PassiveAggressiveClassifier(),
        'RidgeClassifierCV': linear_model.RidgeClassifierCV(),
        'SGDClassifier': linear_model.SGDClassifier(),
        'Perceptron': linear_model.Perceptron(),
        'BernoulliNB': naive_bayes.BernoulliNB(),
        'GaussianNB': naive_bayes.GaussianNB(),
        'KNeighborsClassifier': neighbors.KNeighborsClassifier(),
        'SVC': svm.SVC(probability=True),
        'NuSVC': svm.NuSVC(probability=True),
        'LinearSVC': svm.LinearSVC(),
        'DecisionTreeClassifier': tree.DecisionTreeClassifier(),
        'ExtraTreeClassifier': tree.ExtraTreeClassifier(),
        'LinearDiscriminantAnalysis': discriminant_analysis.LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': discriminant_analysis.QuadraticDiscriminantAnalysis(),
        'XGBClassifier': XGBClassifier(tree_method='hist', device='cuda', eval_metric='mlogloss')
    }

    # Hyperparameter distributions for each model
    param_distributions = {
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0, 1.5]
        },
        'BaggingClassifier': {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 0.8, 1.0],
            'max_features': [0.5, 0.8, 1.0]
        },
        'ExtraTreesClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'GaussianProcessClassifier': {
            'kernel': [None, gaussian_process.kernels.RBF()],
            'n_restarts_optimizer': [0, 1, 5]
        },
        'LogisticRegressionCV': {
            'Cs': [10, 20],
            'cv': [3, 5],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': ['auto', 200, 300, 500, 1000, 10000, 100000, 1000000, 10000000]
        },
        'PassiveAggressiveClassifier': {
            'C': [0.01, 0.1, 1.0],
            'max_iter': [1000, 10000],
            'tol': [1e-3, 1e-4]
        },
        'RidgeClassifierCV': {
            'alphas': [0.1, 1.0, 5.0, 10.0]
        },
        'SGDClassifier': {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2]
        },
        'Perceptron': {
            'alpha': [1e-4, 1e-3, 1e-2],
            'max_iter': [1000, 10000],
            'tol': [1e-3, 1e-4]
        },
        'BernoulliNB': {
            'alpha': [0.01, 0.1, 1.0]
        },
        'GaussianNB': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance']
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        },
        'NuSVC': {
            'nu': [0.1, 0.2, 0.3],
            'kernel': ['linear', 'rbf']
        },
        'LinearSVC': {
            'C': [0.1, 1, 10]
        },
        'DecisionTreeClassifier': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'ExtraTreeClassifier': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'LinearDiscriminantAnalysis': {
            'solver': ['lsqr'],
            'shrinkage': [None, 'auto'] 
        },
        'QuadraticDiscriminantAnalysis': {
            'reg_param': [0.0, 0.1, 0.5]
        },
        'XGBClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    }

    model_names, model_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []
    best_model, best_score = None, 0

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, algo in models.items():
        try:
            if name in param_distributions:
                    search = RandomizedSearchCV(algo, param_distributions[name], n_iter=min(20, len(param_distributions[name])), cv=cv, random_state=42, error_score='raise')
                    search.fit(x_train, y_train)
                    best_model_temp = search.best_estimator_
                    y_pred = best_model_temp.predict(x_test)
            else:
                algo.fit(x_train, y_train)
                y_pred = algo.predict(x_test)

            # Calculate evaluation metrics
            score = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            model_names.append(name)
            model_scores.append(score)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            logging.info(f"{name}: Accuracy = {score:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}")

            if score > best_score:
                best_score = score
                best_model = best_model_temp if 'best_model_temp' in locals() else algo
                logging.info(f"Best Model: {best_model}, Best Parameters: {search.best_params_ if 'search' in locals() else 'N/A'}")

        except Exception as e:
            logging.error(f"Failed to fit {name}: {e}")

    # Model score DataFrame
    model_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': model_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores,
    })

    # Sort by multiple metrics, for example by Accuracy and F1 Score in descending order
    model_df.sort_values(by=['Accuracy', 'F1 Score', 'Recall', 'Precision'], ascending=False, inplace=True)
    logging.info("Model evaluation completed.")
    logging.info(model_df)

    return best_model, model_df

def save_model(model, scaler, le):
    """Save the best model, scaler, and label encoder."""
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    joblib.dump(le, LE_OUTPUT_PATH)

def predict(new_data: np.ndarray) -> np.ndarray:
    """Make predictions with the trained model."""
    loaded_model = joblib.load(MODEL_OUTPUT_PATH)
    loaded_le = joblib.load(LE_OUTPUT_PATH)
    loaded_scaler = joblib.load(SCALER_OUTPUT_PATH)  # Load the scaler here
    scaled_data = loaded_scaler.transform(new_data)  # Scale the new data
    scaled_data = pd.DataFrame(scaled_data, columns=new_data.columns) 
    predictions = loaded_model.predict(scaled_data)
    return loaded_le.inverse_transform(predictions) 

# Main execution
if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    X_scaled, Y, le, scaler = preprocess_data(df)
    best_model, model_df = train_models(X_scaled, Y)
    save_model(best_model, scaler, le)

    # Example usage
    
    # Ensure new_data is a 2D array or DataFrame with proper feature names
    new_data = pd.DataFrame({
      'SepalLengthCm': [2.1],
      'SepalWidthCm': [4.5],
      'PetalLengthCm': [3.4],
      'PetalWidthCm': [1.2]
    })

    logging.info(f"Prediction for new data {new_data}: {predict(new_data)}")
