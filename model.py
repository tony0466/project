import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

def train_model(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Preprocessing
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    X = data.drop(columns=['label'])
    y = data['label']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Isolation Forest for outlier detection
    isolation_forest = IsolationForest(contamination=0.5, random_state=42)
    outliers_train = isolation_forest.fit_predict(X_train)
    outliers_test = isolation_forest.predict(X_test)
    y_train[outliers_train == -1] = 'outlier'
    y_test[outliers_test == -1] = 'outlier'

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, oob_score=True, n_jobs=-1, max_depth=5)
    rf_clf.fit(X_train, y_train)

    return isolation_forest, rf_clf, scaler, X_train, X_test, y_train, y_test

def evaluate_model(isolation_forest, rf_clf, X_test, y_test):
    # Outlier detection evaluation
    outliers_test = isolation_forest.predict(X_test)
    precision_iso = precision_score(y_test == 'outlier', outliers_test == -1)
    recall_iso = recall_score(y_test == 'outlier', outliers_test == -1)
    f1_iso = f1_score(y_test == 'outlier', outliers_test == -1)

    # Random Forest Classifier evaluation
    y_pred = rf_clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)

    return precision_iso, recall_iso, f1_iso, accuracy_rf

def predict_sample(sample_data, isolation_forest, rf_clf, scaler):
    sa_scaled = scaler.transform(sample_data)
    out_prediction = isolation_forest.predict(sa_scaled)
    if out_prediction == -1:
        return "outlier", None
    else:
        class_prediction = rf_clf.predict(sa_scaled)
        return "not an outlier", class_prediction[0]
