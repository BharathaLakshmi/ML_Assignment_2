# Wine Quality Classification – ML Assignment 2

## Problem Statement

Predict whether a wine is of **good quality (1)** or **bad quality (0)** based on
its physicochemical properties (fixed acidity, volatile acidity, alcohol content, etc.).
A wine is labelled *good* if its quality score ≥ 6, otherwise *bad*.
This is a **binary classification** problem solved using five different ML algorithms.

---

## Dataset Description

| Property        | Value                                            |
|-----------------|--------------------------------------------------|
| Source          | Wine Quality dataset (UCI-style)                 |
| Total Instances | 1 500                                            |
| Features        | 12 (physicochemical properties)                  |
| Target Variable | Binary — 0 = Bad quality, 1 = Good quality       |
| Class Balance   | ~50 % Good / ~50 % Bad                           |

### Features

| Feature                | Description                                   |
|------------------------|-----------------------------------------------|
| fixed_acidity          | Tartaric acid concentration (g/L)             |
| volatile_acidity       | Acetic acid (vinegar taste) (g/L)             |
| citric_acid            | Adds freshness (g/L)                          |
| residual_sugar         | Sugar remaining after fermentation (g/L)      |
| chlorides              | Salt content (g/L)                            |
| free_sulfur_dioxide    | Free SO₂ (prevents microbial growth) (mg/L)   |
| total_sulfur_dioxide   | Total SO₂ (mg/L)                              |
| density                | Density of the wine (g/cm³)                   |
| pH                     | Acidity level (scale 0–14)                    |
| sulphates              | Additive contributing to SO₂ (g/L)            |
| alcohol                | Alcohol percentage (% vol)                    |
| wine_type              | 0 = Red wine, 1 = White wine                  |

---

## GitHub Repository Link

> **[https://github.com/YOUR_USERNAME/ml-assignment-2-wine-quality](https://github.com/YOUR_USERNAME/ml-assignment-2-wine-quality)**  
> *(Replace with your actual GitHub URL after pushing)*

---

## Live Streamlit App Link

> **[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**  
> *(Replace with your deployed Streamlit URL)*

---

## Models Used & Evaluation Metrics

All five models were trained on **80%** of the dataset (1 200 rows) and evaluated on
the remaining **20%** (300 rows).

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.7933   | 0.8757 | 0.7722    | 0.8243 | 0.7974 | 0.5883 |
| Decision Tree          | 0.7300   | 0.7479 | 0.7134    | 0.7568 | 0.7344 | 0.4612 |
| KNN                    | 0.7433   | 0.7991 | 0.7320    | 0.7568 | 0.7442 | 0.4871 |
| Naive Bayes            | 0.7733   | 0.8652 | 0.7564    | 0.7973 | 0.7763 | 0.5477 |
| Support Vector Machine | 0.7833   | 0.8581 | 0.7610    | 0.8176 | 0.7883 | 0.5685 |

### Observations on Model Performance

| ML Model Name          | Observation about model performance |
|------------------------|--------------------------------------|
| Logistic Regression    | Best overall performer on this dataset. The relationship between wine physicochemical properties and quality is largely linear (or log-linear), which suits Logistic Regression well. Highest Accuracy (79.33%), AUC (0.8757), F1 (0.7974) and MCC (0.5883). Very stable and interpretable. |
| Decision Tree          | Lowest performance across all metrics. Decision Trees tend to overfit on training data; even with `max_depth=8` the model struggles to generalise. High variance is visible from the relatively large gap between AUC and Accuracy. Feature interactions are not captured as effectively as ensemble methods. |
| KNN                    | Moderate performance. KNN is sensitive to the scale of features (StandardScaler helps) and the choice of k. With k=7, the model achieves decent recall but lags behind Logistic Regression and SVM. It also becomes slow at prediction time as the dataset grows. |
| Naive Bayes            | Better than expected given its strong independence assumption. The Gaussian NB model works reasonably well because many features (pH, alcohol, acidity) approximate Gaussian distributions. Second-best AUC (0.8652) suggests good probabilistic calibration. |
| Support Vector Machine | Strong performance, very close to Logistic Regression. The RBF kernel captures non-linear boundaries effectively. Recall of 0.8176 is the second highest, indicating the model is good at identifying good-quality wines. Computationally more expensive than Logistic Regression. |
| **Overall Winner**     | **Logistic Regression** – highest Accuracy, AUC, F1, and MCC. For this dataset the decision boundary is approximately linear, which plays to Logistic Regression's strengths. It is also the fastest to train and the most interpretable. |

---

## Repository Structure

```
ML_Assignment_2/
├── app.py                  # Streamlit web application
├── train_models.py         # Script to train & save all 5 models
├── wine_quality.csv        # Full dataset (1 500 rows)
├── test_data.csv           # Test split (300 rows) for app upload
├── model_results.csv       # Summary of evaluation metrics
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-assignment-2-wine-quality.git
cd ml-assignment-2-wine-quality

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train models
python train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use the Streamlit App

1. Open the app using the Streamlit link above
2. Go to the **📊 Single Model** tab
3. Upload `test_data.csv` (provided in the repository)
4. Select a model from the dropdown
5. View **Accuracy, AUC, Precision, Recall, F1, MCC** metrics
6. Explore the **Confusion Matrix**, **Classification Report**, and **ROC Curve**
7. Use **📈 Compare All Models** tab to compare all 5 classifiers simultaneously

---

*ML Assignment 2 – M.Tech AIML | Work Integrated Learning Programmes Division | BITS Pilani WILP*
