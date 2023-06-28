from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from src.utils import read_config
from src.net import Net
from src.training import Trainer
from src.dataset import get_dataloaders
from database.mongo import MongoDB
from src.logger import LOGGER
from src import kafka_utils


def predict(net, test_dataloader, emb_db, device, thresh=0.8):
    train_embs, train_labels = torch.stack(list(emb_db.values())), list(emb_db.keys())
    train_embs = train_embs.squeeze(1)
    labels = [[] for _ in range(len(test_dataloader.dataset))]
    with torch.no_grad():
        # work only with batch_size=1 in dataloader!
        for i, data in enumerate(tqdm(test_dataloader)):
            emb = net(data.to(device))
            cos_sims = torch.nn.functional.cosine_similarity(emb, train_embs, dim=1)
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
            "Id": [" ".join(l) for l in map(str, labels)],
        }
    )
    return prediction


def main():
    cfg = read_config(Path(__file__).parent / "config.yml")

    LOGGER.info("Initialize database client")
    db_client = MongoDB()

    LOGGER.info("Initialize dataloaders")
    _, _, test_dl = get_dataloaders(
        Path(cfg.infer.dataset_dir), cfg.infer.dataset_dir.batch_size
    )

    LOGGER.info("Load checkpoint")
    net = Net(cfg.train)
    ckpt_path = cfg.infer.checkpoint_path
    if ckpt_path:
        net.load_state_dict(Trainer(cfg.train).load_ckpt(ckpt_path)["net_state"])
    net.to(cfg.infer.device)
    net.eval()

    LOGGER.info("Uploading embeddings")
    emb_db = db_client.get_all_embeddings()

    LOGGER.info("Prediction")
    prediction = predict(net, test_dl, emb_db, cfg.infer.device, cfg.infer.threshold)

    LOGGER.info("Save predictions")
    db_client.insert_predicts(prediction)

    LOGGER.info("Send predictions to Kafka")
    kafka_utils.send_kafka(prediction)


if __name__ == "__main__":
    main()
