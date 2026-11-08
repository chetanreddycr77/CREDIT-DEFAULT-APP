"""Train 5 classifiers on the UCI Credit Card Default dataset and save them."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "default.payment.next.month"
RANDOM_STATE = 42

df = pd.read_csv("UCI_Credit_Card.csv").drop(columns=["ID"])

# Undocumented category codes -> merge into the 'others' bucket
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Held-out test file that will be uploaded into the Streamlit app
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)),
}

def scores(model, X):
    # AUC needs a continuous score; SVC uses decision_function
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, scores(model, X_test)),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
print(res.to_markdown(index=False))
