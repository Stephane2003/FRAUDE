import json
import time
import joblib
import pandas as pd

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from elasticsearch import Elasticsearch


# ==============================
# CHARGEMENT PIPELINE ML
# ==============================

pipeline = joblib.load("xgboost_fraud_pipeline.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
threshold = pipeline["threshold"]
features = pipeline["features"]


# ==============================
# CONNEXION ELASTICSEARCH
# ==============================

while True:

    try:
        es = Elasticsearch("http://elasticsearch:9200")

        es.info()

        print("Connexion Elasticsearch réussie")

        break

    except Exception as e:

        print("Elasticsearch pas prêt...")
        print(e)

        time.sleep(5)

INDEX_NAME = "fraud-detection"


# ==============================
# CRÉATION INDEX
# ==============================

if not es.indices.exists(index=INDEX_NAME):

    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "fraud_probability": {"type": "float"},
                    "prediction": {"type": "integer"},
                    "true_class": {"type": "integer"},
                    "@timestamp": {"type": "date"}
                }
            }
        }
    )

    print(f"Index '{INDEX_NAME}' créé")


# ==============================
# CONNEXION KAFKA
# ==============================

consumer = None

while consumer is None:

    try:
        consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers='kafka:9092',

            auto_offset_reset='earliest',

            group_id="fraud-group",

            enable_auto_commit=True,

            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        print("Connexion Kafka réussie pour Consumer")

    except NoBrokersAvailable:
        print("Kafka indisponible...")
        time.sleep(5)

print("Consumer démarré...")


# ==============================
# STREAM PROCESSING
# ==============================

for message in consumer:

    try:

        data = message.value

        # vraie classe
        true_class = data.get("Class", 0)

        # suppression colonnes inutiles
        data.pop("Class", None)
        data.pop("Time", None)

        # dataframe
        df = pd.DataFrame([data])

        # ordre colonnes
        df = df.reindex(columns=features)

        # scaling
        df["Amount"] = scaler.transform(df[["Amount"]]).ravel()

        # prédiction
        probability = model.predict_proba(df)[0][1]

        prediction = int(probability >= threshold)

        # affichage
        print("\n==============================")
        print(f"Probabilité fraude : {probability:.4f}")

        if prediction == 1:
            print("ALERTE FRAUDE")
        else:
            print("Transaction normale")

        print(f"Classe réelle : {true_class}")
        print("==============================")

        # ==========================
        # DOCUMENT ELASTICSEARCH
        # ==========================

        elastic_doc = data.copy()

        elastic_doc["fraud_probability"] = float(probability)
        elastic_doc["prediction"] = prediction
        elastic_doc["true_class"] = int(true_class)

        elastic_doc["@timestamp"] = pd.Timestamp.utcnow().isoformat()

        # ==========================
        # INDEXATION
        # ==========================

        response = es.index(
            index=INDEX_NAME,
            document=elastic_doc
        )

        print(f"Document indexé : {response['_id']}")

    except Exception as e:
        print("Erreur traitement message :", e)