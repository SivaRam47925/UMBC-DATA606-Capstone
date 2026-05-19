import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load final dataset
df = pd.read_csv("final_16_features_plus_target.csv")

target = "hospital_death"

X = df.drop(columns=[target])
y = df[target]

# Save exact feature order
features = list(X.columns)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-Score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

# Save artifacts
joblib.dump(model, "final_gradient_boosting_model.pkl")
joblib.dump(imputer, "model_imputer.pkl")
joblib.dump(features, "model_features.pkl")

print("\nSaved:")
print("- final_gradient_boosting_model.pkl")
print("- model_imputer.pkl")
print("- model_features.pkl")
