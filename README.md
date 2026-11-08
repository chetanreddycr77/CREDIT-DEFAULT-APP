# Credit Card Default Prediction – ML Assignment 2 (M.Tech AIML, BITS Pilani WILP)

## a. Problem statement
Predict whether a credit-card customer in Taiwan will **default on their payment next month** (binary classification: 1 = default, 0 = no default), using six months of repayment status, bill amounts, payment amounts and demographic information. Five classifiers are trained on the same data, compared on six metrics, and exposed through a Streamlit web app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository – *Default of Credit Card Clients* (Kaggle copy: `UCI_Credit_Card.csv`)
- **Instances:** 30,000  |  **Features:** 23 (after dropping `ID`)  |  **Target:** `default.payment.next.month`
- **Class balance:** 77.9% no default / 22.1% default (imbalanced)
- **Features:** `LIMIT_BAL` (credit limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0, PAY_2–PAY_6` (repayment status, Apr–Sep), `BILL_AMT1–6` (bill statements), `PAY_AMT1–6` (previous payments)
- **Preprocessing:** dropped `ID`; undocumented `EDUCATION` codes (0, 5, 6) merged into 4 ("others"); `MARRIAGE` = 0 merged into 3; no missing values. `StandardScaler` is inside the pipelines of LR, KNN, NB and SVM. Stratified 80/20 train/test split (`random_state=42`); the 6,000-row test split is saved as `test_data.csv`.

## c. GitHub repository link
<PASTE YOUR GITHUB REPO URL HERE>

Live app: <PASTE YOUR STREAMLIT URL HERE>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8083 | 0.7079 | 0.6903 | 0.2419 | 0.3583 | 0.3276 |
| Decision Tree | 0.8172 | 0.7417 | 0.6624 | 0.3534 | 0.4609 | 0.3889 |
| KNN | 0.8062 | 0.7319 | 0.6152 | 0.3301 | 0.4296 | 0.3483 |
| Naive Bayes (Gaussian) | 0.7518 | 0.7248 | 0.4504 | 0.5539 | 0.4968 | 0.3376 |
| Support Vector Machine (RBF) | 0.8162 | 0.7186 | 0.6662 | 0.3384 | 0.4488 | 0.3814 |

(Majority-class baseline accuracy on this test set = 0.779. Metrics are for the default class = 1.)

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is decent and precision is high (0.69), but recall is only 0.24: it misses about three out of four defaulters. A linear boundary cannot capture the non-linear effect of repayment delays, giving the lowest AUC (0.708). |
| Decision Tree | Best accuracy, AUC and MCC. Depth limited to 5 with `min_samples_leaf=50` to avoid overfitting. It relies mostly on `PAY_0` (about 68% of importance) and `PAY_2`, which matches the intuition that recent delinquency predicts default. Also the most interpretable. |
| KNN (k=15) | Middle-of-the-pack on every metric (AUC 0.732). Works reasonably after scaling, but 24 features including many correlated bill amounts dilute the distance metric, and prediction on the test set is slower than other models. |
| Naive Bayes | Lowest accuracy (0.752, below the majority baseline) because the independence assumption is violated by the highly correlated `BILL_AMT` columns. But it has the highest recall (0.554) and F1 (0.497), as it flags many more customers as risky. Useful if catching defaulters matters more than false alarms. |
| Support Vector Machine | Very close to the Decision Tree on accuracy (0.816) and MCC (0.381) with higher AUC than LR, but lower than DT and KNN. Slowest to train and larger to store, with no clear gain over the tree. |
| **Overall winner** | **Decision Tree**: best Accuracy, AUC and MCC and second-best F1, with a small model size and interpretable rules. If recall for defaulters is the priority, Naive Bayes is the better pick. |

## How to run
```bash
pip install -r requirements.txt
python train_models.py        # retrains models, writes model/*.joblib and test_data.csv
streamlit run app.py
```
Then upload `test_data.csv` in the app sidebar, pick a model, and view metrics, confusion matrix, ROC curve and classification report.

## Repository structure
```
app.py  |  train_models.py  |  requirements.txt  |  README.md  |  test_data.csv
UCI_Credit_Card.csv  |  model/ (5 × .joblib + metrics_summary.csv)
```
