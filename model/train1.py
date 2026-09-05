# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

# ==============================
# 2. CHARGEMENT DATASET
# ==============================
df = pd.read_csv("data/creditcard.csv")

X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]

# ==============================
# 3. SPLIT
# ==============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ==============================
# 4. SCALING (important pour stabilité)
# ==============================
scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_val["Amount"] = scaler.transform(X_val[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# ==============================
# 5. GESTION DU DÉSÉQUILIBRE
# ==============================
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# ==============================
# 6. MODÈLE XGBOOST
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# ==============================
# 7. ENTRAÎNEMENT
# ==============================
model.fit(X_train, y_train)

print(" Modèle XGBoost entraîné avec succès")

# ==============================
# 8. PRÉDICTIONS (probabilités)
# ==============================
val_probs = model.predict_proba(X_val)[:, 1]

# ==============================
# 9. CHOIX DU SEUIL (F1 OPTIMAL)
# ==============================
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n Seuil optimal : {best_threshold:.4f}")

# ==============================
# 10. TEST FINAL
# ==============================
test_probs = model.predict_proba(X_test)[:, 1]
y_pred = (test_probs >= best_threshold).astype(int)

# ==============================
# 11. ÉVALUATION
# ==============================
print("\n RAPPORT FINAL :")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matrice de Confusion - XGBoost")
plt.show()

# ==============================
# 12. MÉTRIQUES GLOBALES
# ==============================
roc_auc = roc_auc_score(y_test, test_probs)
ap = average_precision_score(y_test, test_probs)

print("\n MÉTRIQUES :")
print(f"AUC ROC           : {roc_auc:.4f}")
print(f"Average Precision : {ap:.4f}")
print(f"Fraudes détectées : {sum(y_pred)}")
print(f"Fraudes réelles   : {sum(y_test)}")

# ==============================
# 13. SAUVEGARDE PIPELINE
# ==============================
pipeline = {
    "model": model,
    "scaler": scaler,
    "threshold": float(best_threshold),
    "features": list(X.columns)
}

joblib.dump(pipeline, "xgboost_fraud_pipeline.pkl")

print("\n Pipeline XGBoost sauvegardé")