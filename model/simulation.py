import pandas as pd
import time
import joblib
import numpy as np


# 1. CHARGEMENT DU DATASET

df = pd.read_csv("fraudes_only.csv")


# 2. CHARGEMENT PIPELINE XGBOOST

pipeline = joblib.load("xgboost_fraud_pipeline.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
threshold = pipeline["threshold"]
features = pipeline["features"]


# 3. PRÉPARATION DES DONNÉES


# garder la vérité terrain pour affichage
y_true = df["Class"]

# suppression colonnes inutiles
X = df.drop(["Class", "Time"], axis=1, errors="ignore").copy()

# garder uniquement les features utilisées
X = X[features]

# normalisation de Amount
X["Amount"] = scaler.transform(X[["Amount"]]).ravel()


# 4. SIMULATION TEMPS RÉEL

print("=" * 60)
print("SIMULATION TEMPS RÉEL - DÉTECTION DE FRAUDE")
print("=" * 60)

fraudes_detectees = 0
transactions_total = 0


# 5. BOUCLE STREAMING

for i, row in X.iterrows():

    transactions_total += 1

  
    # Préparation transaction

    data = row.values.reshape(1, -1)


    # Probabilité de fraude

    fraud_probability = model.predict_proba(data)[0][1]


    # Décision selon seuil

    prediction = int(fraud_probability >= threshold)


    # Résultat affichage

    if prediction == 1:
        result = "FRAUDE"
        fraudes_detectees += 1
    else:
        result = "NORMAL"

    # vérité terrain
    real_class = "FRAUDE" if y_true.iloc[i] == 1 else "NORMAL"


    # AFFICHAGE

    print(f"\nTransaction #{i}")
    print(f"Montant           : {row['Amount']:.4f}")
    print(f"Probabilité fraude: {fraud_probability:.4f}")
    print(f"Prédiction modèle : {result}")
    print(f"Classe réelle     : {real_class}")

    # vérification
    if prediction == y_true.iloc[i]:
        print("Statut            : CORRECT")
    else:
        print("Statut            : ERREUR")

    print("-" * 60)


    # PAUSE STREAMING

    time.sleep(0.5)

# 6. STATISTIQUES FINALES

print("\n")
print("=" * 60)
print("FIN DE LA SIMULATION")
print("=" * 60)

print(f"Transactions traitées : {transactions_total}")
print(f"Fraudes détectées     : {fraudes_detectees}")
print(f"Fraudes réelles       : {sum(y_true)}")
