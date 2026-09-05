import pandas as pd
import json
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ==============================
# CONNEXION KAFKA AVEC RETRY
# ==============================

producer = None

while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers='kafka:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        print("Connexion Kafka réussie pour Producer")

    except NoBrokersAvailable:
        print("Kafka indisponible... nouvelle tentative dans 5 secondes")
        time.sleep(5)

# ==============================
# CHARGEMENT DATASET
# ==============================

df = pd.read_csv("data/fraudes_melange.csv")

# ==============================
# ENVOI STREAMING
# ==============================

print("Producer démarré...")

sample_df = df.sample(200, random_state=42)

for i, row in sample_df.iterrows():

    data = row.to_dict()

    producer.send("transactions", value=data)

    print(f"Transaction envoyée : {i}")

    time.sleep(20)

producer.flush()