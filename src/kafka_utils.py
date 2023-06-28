import json
import os
import pickle

import pandas as pd
from ansible_vault import Vault
from kafka import KafkaProducer, KafkaConsumer


ANSIBLE_PASSWD = os.environ.get("ANSIBLE_PASSWD")
vault = Vault(ANSIBLE_PASSWD)
with open("kafka.credentials", "r") as f:
    credentials = iter(vault.load(f.read()).split(" "))
KAFKAHOST = next(credentials)
KAFKAPORT = next(credentials)

PREDICT_TOPIC = "kafka-pred"
KAFKA_PRODUCER = KafkaProducer(
    bootstrap_servers=f"{KAFKAHOST}:{KAFKAPORT}",
    value_serializer=lambda v: v.to_json().encode("utf-8"),
)
KAFKA_CONSUMER = KafkaConsumer(
    PREDICT_TOPIC,
    bootstrap_servers=f"{KAFKAHOST}:{KAFKAPORT}",
    value_deserializer=lambda x: pd.DataFrame(json.loads(x.decode("utf-8"))),
    fetch_max_wait_ms=300_000,  # 5 minutes
    auto_offset_reset="earliest",
)


def send_kafka(predictions: pd.DataFrame):
    predictions.to_csv("tmp.csv", index=False)
    predictions = pd.read_csv("tmp.csv", encoding="unicode_escape")
    os.remove("tmp.csv")
    KAFKA_PRODUCER.send(PREDICT_TOPIC, predictions)


def get_prediction_from_kafka() -> pd.DataFrame:
    prediction = next(KAFKA_CONSUMER).value
    return prediction
