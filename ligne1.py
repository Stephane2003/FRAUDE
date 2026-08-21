import pandas as pd

# ==============================
# CHARGEMENT DATASET
# ==============================
df = pd.read_csv("data/creditcard.csv")

# ==============================
# FILTRER LES DONNÉES
# ==============================

# Sélectionner 80 lignes où Class = 1 (Fraudes)
fraudes = df[df["Class"] == 1].head(80)

# Sélectionner 120 lignes où Class = 0 (Légitimes)
legitimes = df[df["Class"] == 0].head(120)

# Combiner les deux sélections
df_final = pd.concat([fraudes, legitimes])

# Optionnel : Mélanger les lignes pour ne pas avoir les fraudes groupées au début
df_final = df_final.sample(frac=1).reset_index(drop=True)

# ==============================
# AFFICHAGE & SAUVEGARDE
# ==============================
print(f"Nombre de fraudes sélectionnées : {len(fraudes)}")
print(f"Nombre de transactions légitimes sélectionnées : {len(legitimes)}")
print(f"Total des lignes à enregistrer : {len(df_final)}")

# Sauvegarde dans le fichier CSV
df_final.to_csv("fraudes_melange.csv", index=False)

print("\nAnalyse terminée. Fichier 'fraudes_melange.csv' sauvegardé.")