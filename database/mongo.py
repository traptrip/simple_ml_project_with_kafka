import os

import pandas as pd
import torch
from torch import Tensor
from pymongo import MongoClient
from ansible_vault import Vault


def get_credentials():
    ANSIBLE_PASSWD = os.environ.get("ANSIBLE_PASSWD")
    vault = Vault(ANSIBLE_PASSWD)
    with open("db.credentials", "r") as f:
        credentials = iter(vault.load(f.read()).split(" "))
    DB_USER = next(credentials)
    DB_PASSWD = next(credentials)
    DB_HOST = next(credentials)
    DB_PORT = next(credentials)
    DB_NAME = next(credentials)
    return DB_USER, DB_PASSWD, DB_HOST, DB_PORT, DB_NAME


class MongoDB:
    def __init__(self):
        DB_USER, DB_PASSWD, DB_HOST, DB_PORT, DB_NAME = get_credentials()
        self.url = f"mongodb://{DB_USER}:{DB_PASSWD}@{DB_HOST}:{DB_PORT}/"
        self.mongo = MongoClient(self.url)
        self.db = self.mongo[DB_NAME]
        self.predicts_collection = self.db["predict"]
        self.embeddings_collection = self.db["embeddings"]

    def get_all_embeddings(self):
        embeddings = list(self.embeddings_collection.find())
        embeddings = {e["Id"]: torch.tensor(e["emb"])[None] for e in embeddings}
        return embeddings

    def get_all_predicts(self):
        predicts = [
            {"Image": p["Image"], "Id": " ".join(p["Id"])}
            for p in list(self.predicts_collection.find())
        ]
        return pd.DataFrame(predicts)

    def insert_embeddings(self, embeddings: dict[str, Tensor]):
        embs_list = [
            {"Id": int(_id), "emb": emb.flatten().tolist()}
            for _id, emb in embeddings.items()
        ]
        self.embeddings_collection.insert_many(embs_list)

    def insert_predicts(self, predicts_df: pd.DataFrame):
        predicts = [
            {"Image": row[0], "Id": row[1].split()} for row in predicts_df.itertuples()
        ]
        self.predicts_collection.insert_many(predicts)
