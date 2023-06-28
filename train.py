import logging
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.utils import read_config
from src.training import Trainer
from database.mongo import MongoDB
from src.logger import LOGGER


def init_embeddings_db(train_dl, net, emb_size, batch_size, device, db_client: MongoDB):
    net.to(device)
    net.eval()
    with torch.no_grad():
        embeddings = torch.zeros((len(train_dl.dataset), emb_size))
        labels = []
        for i, (batch, label) in enumerate(tqdm(train_dl, desc="Init embeddings DB")):
            labels.extend(label.tolist())
            embeddings[i : i + batch_size] = net(batch.to(device)).cpu()

        # generate embeddings database for each label
        emb_db = defaultdict(list)
        id2label = train_dl.dataset.id2label
        for i, id_ in enumerate(labels):
            emb_db[id2label[id_]].append(embeddings[i])

        # average embeddings for each label
        for label, embs in emb_db.items():
            emb_db[label] = torch.nn.functional.normalize(
                torch.mean(torch.stack(embs), dim=0, keepdims=True)
            )

    db_client.insert_embeddings(emb_db)


if __name__ == "__main__":
    cfg = read_config(Path(__file__).parent / "config.yml")

    # prepare database
    LOGGER.info("Initialize database client")
    db_client = MongoDB()

    # prepare data
    LOGGER.info("Prepare dataloaders")
    train_dl, val_dl, _ = get_dataloaders(Path(cfg.data.data_dir), cfg.train.batch_size)

    LOGGER.info("Training")
    trainer = Trainer(cfg.train)
    trainer.train(train_dl, val_dl)

    # prepare embeddings_file
    LOGGER.info("Insert embeddings into DB")
    init_embeddings_db(
        train_dl,
        trainer.net,
        cfg.train.net.embedding_size,
        cfg.train.batch_size,
        cfg.train.device,
        db_client,
    )
