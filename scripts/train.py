import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load dataset (path relative to this script)
from pathlib import Path

data_path = Path(__file__).resolve().parent / "data" / "telco_churn.csv.csv"
df = pd.read_csv(data_path)

# 2. Basic preprocessing (example: drop customerID, encode target)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical columns to dummy variables
df = pd.get_dummies(df)

# 3. Split features/labels
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost model
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

# 6. Log with MLflow
with mlflow.start_run():
    # Ensure estimator type is set for MLflow/xgboost save_model on some versions
    model._estimator_type = "classifier"
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)
    mlflow.xgboost.log_model(model, name="model")

print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
