import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

TARGET = "default.payment.next.month"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.set_page_config(page_title="Credit Card Default Predictor", page_icon="💳", layout="wide")
st.title("💳 Credit Card Default – Model Evaluation Dashboard")
st.caption("Predict whether a customer will default next month (UCI Taiwan Credit Card dataset). "
           "Models are pre-trained; the uploaded file is used only for evaluation.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_scores(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def clean(df):
    df = df.drop(columns=["ID"], errors="ignore").copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type="csv")
    choice = st.selectbox("Choose a model", list(MODEL_FILES))
    show_all = st.checkbox("Also compare all 5 models", value=True)
    st.markdown("Target column: `default.payment.next.month` (1 = default)")

if uploaded is None:
    st.info("⬅️ Upload `test_data.csv` from the GitHub repo to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"Target column `{TARGET}` not found in the uploaded file.")
    st.stop()

data = clean(data)
X, y = data.drop(columns=[TARGET]), data[TARGET]
st.subheader("Uploaded data")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Features", X.shape[1])
c3.metric("Default rate", f"{y.mean():.1%}")
with st.expander("Preview"):
    st.dataframe(data.head(20))


def evaluate(name):
    model = load_model(MODEL_FILES[name])
    pred = model.predict(X)
    return model, pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, get_scores(model, X)),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


model, pred, metrics = evaluate(choice)

st.subheader(f"Results – {choice}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["No default", "Default"], cmap="Blues", ax=ax)
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y, get_scores(model, X))
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("**Classification report**")
st.dataframe(pd.DataFrame(classification_report(
    y, pred, target_names=["No default", "Default"], output_dict=True)).T.round(3))

if show_all:
    st.subheader("All models on the uploaded data")
    table = pd.DataFrame({n: evaluate(n)[2] for n in MODEL_FILES}).T.round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf"))
    st.bar_chart(table[["AUC", "F1", "MCC"]])
