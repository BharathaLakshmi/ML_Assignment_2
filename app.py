"""
app.py  –  Wine Quality Classifier  |  ML Assignment 2
=======================================================
Streamlit app with:
  • Dataset upload (CSV)
  • Model selection dropdown
  • Evaluation metrics display
  • Confusion matrix & classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    RocCurveDisplay,
)

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "Logistic Regression":    "logistic_regression.joblib",
    "Decision Tree":          "decision_tree.joblib",
    "KNN":                    "knn.joblib",
    "Naive Bayes":            "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

TARGET_COL = "target"

# ── Helpers ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    path = os.path.join(BASE_DIR, "model", MODEL_FILES[model_name])
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),               4),
        "AUC":       round(roc_auc_score(y_true, y_proba),               4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred),                  4),
        "F1 Score":  round(f1_score(y_true, y_pred),                      4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred),             4),
    }


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Bad (0)", "Good (1)"],
        yticklabels=["Bad (0)", "Good (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Wine_grapes_on_the_vine.jpg/640px-Wine_grapes_on_the_vine.jpg",
    use_column_width=True,
)
st.sidebar.title("🍷 Wine Quality Classifier")
st.sidebar.markdown(
    "**Dataset:** Synthetic Wine Quality  \n"
    "**Task:** Binary Classification  \n"
    "*(Good wine: quality ≥ 6)*"
)

# ── Main header ──────────────────────────────────────────────────────────────────
st.title("🍷 Wine Quality Classification – ML Assignment 2")
st.markdown(
    "Upload the **test CSV** (same format as `test_data.csv`), "
    "choose a model, and explore evaluation metrics."
)

# ── Tab layout ───────────────────────────────────────────────────────────────────
tab_single, tab_compare, tab_about = st.tabs(
    ["📊 Single Model", "📈 Compare All Models", "ℹ️ About"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single Model Evaluation
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_up, col_sel = st.columns([2, 1])

    with col_up:
        st.subheader("1️⃣  Upload Test Data (CSV)")
        uploaded = st.file_uploader(
            "Upload CSV with features + 'target' column",
            type="csv",
            key="single_upload",
        )

    with col_sel:
        st.subheader("2️⃣  Select Model")
        model_name = st.selectbox(
            "Choose a classifier",
            list(MODEL_FILES.keys()),
            key="single_model",
        )

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Validate
        missing_feats = [f for f in FEATURES if f not in df.columns]
        if missing_feats:
            st.error(f"Missing columns in uploaded file: {missing_feats}")
            st.stop()
        if TARGET_COL not in df.columns:
            st.error(f"Column '{TARGET_COL}' not found. Upload file must include the target column.")
            st.stop()

        X = df[FEATURES]
        y = df[TARGET_COL]

        st.success(f"Loaded {len(df)} rows  |  {X.shape[1]} features")

        with st.expander("📋 Preview uploaded data"):
            st.dataframe(df.head(10), use_container_width=True)

        # Predict
        model = load_model(model_name)
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_pred, y_proba)
        cm      = confusion_matrix(y, y_pred)

        st.markdown("---")
        st.subheader(f"3️⃣  Results  –  {model_name}")

        # Metric cards
        mcols = st.columns(6)
        icons = ["✅", "📐", "🎯", "🔍", "⚖️", "🏅"]
        for col, (k, v), icon in zip(mcols, metrics.items(), icons):
            col.metric(f"{icon} {k}", v)

        st.markdown("---")
        col_cm, col_rep = st.columns([1, 2])

        with col_cm:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(cm, model_name)
            st.pyplot(fig)

        with col_rep:
            st.subheader("Classification Report")
            report = classification_report(
                y, y_pred,
                target_names=["Bad Wine (0)", "Good Wine (1)"],
                output_dict=True,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                use_container_width=True,
            )

        # ROC Curve
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(
            y, y_proba,
            name=model_name,
            ax=ax_roc,
        )
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
        ax_roc.set_title(f"ROC Curve – {model_name}")
        ax_roc.legend()
        plt.tight_layout()
        st.pyplot(fig_roc)

    else:
        st.info("👆 Upload `test_data.csv` to begin. You can find this file in the GitHub repository.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Compare All Models
# ════════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare All 5 Classifiers")
    uploaded_cmp = st.file_uploader(
        "Upload CSV with features + 'target' column",
        type="csv",
        key="compare_upload",
    )

    if uploaded_cmp is not None:
        df_cmp = pd.read_csv(uploaded_cmp)
        missing = [f for f in FEATURES if f not in df_cmp.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        if TARGET_COL not in df_cmp.columns:
            st.error("'target' column missing.")
            st.stop()

        X_cmp = df_cmp[FEATURES]
        y_cmp = df_cmp[TARGET_COL]

        all_metrics = []
        all_cms     = {}

        for mname in MODEL_FILES:
            m      = load_model(mname)
            yp     = m.predict(X_cmp)
            yproba = m.predict_proba(X_cmp)[:, 1]
            row    = {"Model": mname}
            row.update(compute_metrics(y_cmp, yp, yproba))
            all_metrics.append(row)
            all_cms[mname] = confusion_matrix(y_cmp, yp)

        summary = pd.DataFrame(all_metrics).set_index("Model")

        st.subheader("📊 Comparison Table")
        st.dataframe(
            summary.style.highlight_max(axis=0, color="lightgreen")
                         .highlight_min(axis=0, color="lightyellow"),
            use_container_width=True,
        )

        # Bar chart
        st.subheader("📈 Metric Bar Chart")
        metric_choice = st.selectbox(
            "Select metric to visualise",
            ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        )
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        colors = sns.color_palette("husl", len(summary))
        bars = ax_bar.bar(summary.index, summary[metric_choice], color=colors)
        ax_bar.bar_label(bars, fmt="%.4f", padding=3)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel(metric_choice)
        ax_bar.set_title(f"{metric_choice} Comparison")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig_bar)

        # Confusion matrices
        st.subheader("🔲 Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, (mname, cm) in enumerate(all_cms.items()):
            with cm_cols[idx % 3]:
                fig_cm = plot_confusion_matrix(cm, mname)
                st.pyplot(fig_cm)

        # Winner
        winner = summary["Accuracy"].idxmax()
        st.success(f"🏆 **Winner (by Accuracy):** {winner}  —  Accuracy = {summary.loc[winner, 'Accuracy']}")
    else:
        st.info("👆 Upload `test_data.csv` to compare all models side by side.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – About
# ════════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About This App")
    st.markdown("""
### Problem Statement
Predict whether a wine is of **good quality (1)** or **bad quality (0)** based on
its physicochemical properties.  A wine is labelled *good* if its quality score ≥ 6.

### Dataset Description
| Property | Value |
|---|---|
| Source | Wine Quality (UCI-style, synthetic) |
| Instances | 1 500 |
| Features | 12 |
| Target | Binary (0 = Bad, 1 = Good) |
| Class balance | ~50 / 50 |

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free SO₂, total SO₂, density, pH, sulphates, alcohol, wine type (0=red/1=white)

### Models Implemented
| # | Model |
|---|---|
| 1 | Logistic Regression |
| 2 | Decision Tree Classifier |
| 3 | K-Nearest Neighbor (KNN) |
| 4 | Naive Bayes (Gaussian) |
| 5 | Support Vector Machine (RBF kernel) |

### Evaluation Metrics
Accuracy · AUC · Precision · Recall · F1 Score · Matthews Correlation Coefficient (MCC)

### How to Use
1. Go to the **📊 Single Model** tab
2. Upload `test_data.csv` (from the GitHub repo)
3. Select a classifier from the dropdown
4. View metrics, confusion matrix, classification report and ROC curve

Use the **📈 Compare All Models** tab to see all 5 models side by side.
""")
    st.markdown("---")
    st.caption("ML Assignment 2 – M.Tech AIML | Work Integrated Learning Programmes Division")
