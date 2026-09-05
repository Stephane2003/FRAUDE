# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
)

# ==============================
# 2. CONFIG
# ==============================
RANDOM_STATE = 42
TARGET_RECALL = 0.60      # plus réaliste
MIN_PRECISION = 0.10      # éviter explosion faux positifs

# ==============================
# 3. CHARGEMENT
# ==============================
df = pd.read_csv("data/creditcard.csv")

X = df.drop(["Class", "Time"], axis=1, errors='ignore').copy()
y = df["Class"]

X = X.astype(float)

# ==============================
# 4. SPLIT
# ==============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)

# ==============================
# 5. SCALING
# ==============================
scaler = StandardScaler()

for dataset in [X_train, X_val, X_test]:
    dataset.loc[:, "Amount"] = dataset["Amount"].astype(float)

X_train.loc[:, "Amount"] = scaler.fit_transform(X_train[["Amount"]]).ravel()
X_val.loc[:, "Amount"] = scaler.transform(X_val[["Amount"]]).ravel()
X_test.loc[:, "Amount"] = scaler.transform(X_test[["Amount"]]).ravel()

# ==============================
# 6. ENTRAÎNEMENT
# ==============================
X_train_normal = X_train[y_train == 0]

model = IsolationForest(
    n_estimators=200,
    contamination=0.0017,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_normal)
print(" Modèle entraîné (normal only)")

# ==============================
# 7. SEUIL ROBUSTE (PR + CONTRAINTES)
# ==============================
val_scores = -model.decision_function(X_val)

precision, recall, thresholds = precision_recall_curve(y_val, val_scores)

best_threshold = None

# ⚠️ attention: thresholds = len(precision) - 1
for i in range(len(thresholds)):
    p = precision[i]
    r = recall[i]
    t = thresholds[i]

    if r >= TARGET_RECALL and p >= MIN_PRECISION:
        best_threshold = t
        print(f" Seuil trouvé avec contraintes : recall={r:.2f}, precision={p:.2f}")
        break

# fallback robuste
if best_threshold is None:
    best_threshold = np.percentile(val_scores, 99.7)
    print(" Aucun seuil valide  fallback percentile 99.5")

print(f"Seuil final : {best_threshold:.5f}")

# ==============================
# 8. TEST
# ==============================
test_scores = -model.decision_function(X_test)
y_test_pred = (test_scores >= best_threshold).astype(int)

# ==============================
# 9. ÉVALUATION
# ==============================
print("\n RAPPORT FINAL :")
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matrice de Confusion")
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, test_scores)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# ROC
fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc = auc(fpr, tpr)

ap = average_precision_score(y_test, test_scores)

print("\n MÉTRIQUES FINALES :")
print(f"AUC ROC           : {roc_auc:.4f}")
print(f"Average Precision : {ap:.4f}")
print(f"Fraudes réelles   : {int(sum(y_test))}")
print(f"Fraudes détectées : {int(sum(y_test_pred))}")

# ==============================
# 10. ERREURS
# ==============================
false_positives = int(np.sum((y_test == 0) & (y_test_pred == 1)))
false_negatives = int(np.sum((y_test == 1) & (y_test_pred == 0)))

print("\n ANALYSE ERREURS :")
print(f"Faux positifs  : {false_positives}")
print(f"Faux négatifs  : {false_negatives}")

# ==============================
# 11. SAUVEGARDE
# ==============================
pipeline = {
    "model": model,
    "scaler": scaler,
    "threshold": float(best_threshold),
    "features": list(X.columns)
}

joblib.dump(pipeline, "fraud_detection_pipeline.pkl")

print("\n Pipeline sauvegardé (production-ready)")