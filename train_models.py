"""
train_models.py
---------------
Trains 5 classification models on the Wine Quality dataset and saves them.
Run this script once to generate the model/*.joblib files and test_data.csv.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "wine_quality.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
TEST_PATH  = os.path.join(BASE_DIR, "test_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & split ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["target"])
y  = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test split (features + target together) for Streamlit upload
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(TEST_PATH, index=False)
print(f"Test data saved → {TEST_PATH}  ({len(test_df)} rows)")

# ── Model definitions ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

model_filenames = {
    "Logistic Regression":    "logistic_regression.joblib",
    "Decision Tree":          "decision_tree.joblib",
    "KNN":                    "knn.joblib",
    "Naive Bayes":            "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ── Train, evaluate, save ──────────────────────────────────────────────────────
results = []

for name, pipeline in models.items():
    print(f"\nTraining: {name} …")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "AUC":       round(roc_auc_score(y_test, y_proba),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1":        round(f1_score(y_test, y_pred),        4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print("  ", metrics)

    # Save
    fname = os.path.join(MODEL_DIR, model_filenames[name])
    joblib.dump(pipeline, fname)
    print(f"  Saved → {fname}")

# ── Summary table ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).set_index("Model")
print("\n\n===== Evaluation Summary =====")
print(results_df.to_string())
results_df.to_csv(os.path.join(BASE_DIR, "model_results.csv"))
print("\nResults saved → model_results.csv")
