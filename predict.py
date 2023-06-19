import os
import logging
import pickle
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from ansible_vault import Vault
from kafka import KafkaProducer

from src.utils import read_config
from src.net import Net
from src.training import Trainer
from src.dataset import get_dataloaders
from database.mongo import MongoDB

ANSIBLE_PASSWD = os.environ.get("ANSIBLE_PASSWD")


def send_kafka(predictions):
    vault = Vault(ANSIBLE_PASSWD)
    with open("db.credentials", "r") as f:
        credentials = iter(vault.load(f.read()).split(" "))
    KAFKA_HOST = next(credentials)
    KAFKA_PORT = next(credentials)

    with KafkaProducer(
        bootstrap_servers=f"{KAFKA_HOST}:{KAFKA_PORT}", api_version=(0, 10, 2)
    ) as producer:
        producer.send("kafka-pred", pickle.dumps(predictions))


def predict(net, test_dataloader, emb_db, device, thresh=0.8):
    train_embs, train_labels = torch.stack(list(emb_db.values())), list(emb_db.keys())
    train_embs = train_embs.squeeze(1)
    labels = [[] for _ in range(len(test_dataloader.dataset))]
    with torch.no_grad():
        # work only with batch_size=1 in dataloader!
        for i, data in enumerate(tqdm(test_dataloader)):
            emb = net(data.to(device))
            cos_sims = torch.nn.functional.cosine_similarity(emb, train_embs, dim=0)
            for j, cs in enumerate(cos_sims):
                if cs > thresh:
                    labels[i].append(train_labels[j])

    # filter empty
    for i, label in enumerate(labels):
        if not label:
            labels[i] = ["new_whale"]

    prediction = pd.DataFrame(
        {
            "Image": [img_p.name for img_p in test_dataloader.dataset.imgs_list],
            "Id": [" ".join(l) for l in labels],
        }
    )
    return prediction


if __name__ == "__main__":
    cfg = read_config(Path(__file__).parent / "config.yml")

    logging.info("Initialize database client")
    db_client = MongoDB()

    logging.info("Initialize dataloaders")
    _, _, test_dl = get_dataloaders(Path(cfg.infer.dataset_dir), 1)

    logging.info("Load checkpoint")
    ckpt_path = Path(cfg.infer.checkpoint_path)
    net = Net(cfg.train)
    net.load_state_dict(Trainer(cfg.train).load_ckpt(ckpt_path)["net_state"])
    net.to(cfg.infer.device)
    net.eval()

    logging.info("Uploading embeddings")
    emb_db = db_client.get_all_embeddings()

    logging.info("Prediction")
    prediction = predict(net, test_dl, emb_db, cfg.infer.device, cfg.infer.threshold)

    logging.info("Save predictions")
    db_client.insert_predicts(prediction)

    logging.info("Send predictions to Kafka")
    send_kafka(prediction)
