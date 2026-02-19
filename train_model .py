import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- LOAD DATA ----------
df = pd.read_csv("campus.csv")

# ---------- FEATURE ENGINEERING ----------
df["cgpa"] = df[["ssc_p","hsc_p","degree_p","mba_p"]].mean(axis=1) / 10
df["internship"] = df["workex"].apply(lambda x: 1 if x=="Yes" else 0)
df["communication"] = df["etest_p"] / 10
df["placed"] = df["status"].apply(lambda x: 1 if x=="Placed" else 0)

# Optional skill match feature
np.random.seed(42)
df["skill_match"] = np.random.uniform(20, 100, len(df))

X = df[["cgpa","internship","communication","skill_match"]]
y = df["placed"]

# ---------- TRAIN TEST ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODELS ----------
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ]),
    
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    
    "GradientBoost": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_score = 0

print("\n===== MODEL COMPARISON =====")

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    score = accuracy_score(y_test, preds)
    
    print(f"{name} Accuracy:", round(score, 4))
    
    if score > best_score:
        best_score = score
        best_model = m

print("\n===== BEST MODEL REPORT =====")
final_preds = best_model.predict(X_test)
print(classification_report(y_test, final_preds))

joblib.dump(best_model, "placement_model.pkl")
print("\nBest model saved with accuracy:", round(best_score,4))

# ---------- FEATURE IMPORTANCE ----------
import matplotlib.pyplot as plt
import numpy as np

try:
    # If model is inside Pipeline
    if hasattr(best_model, "named_steps"):
        coef = best_model.named_steps["model"].coef_[0]
    else:
        coef = best_model.coef_[0]

    features = X.columns

    plt.figure(figsize=(6,4))
    plt.barh(features, coef)
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.xlabel("Impact on Placement Prediction")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Feature importance not available:", e)

