import pytest

from predict import predict
from train import init_embeddings_db
from database.mongo import MongoDB
from src import kafka_utils


@pytest.mark.slow
def test_predict(config, model, dataloaders):
    db_client = MongoDB()
    init_embeddings_db(
        dataloaders["train"],
        model,
        config.train.net.embedding_size,
        config.train.batch_size,
        config.train.device,
        db_client,
    )

    emb_db = db_client.get_all_embeddings()
    assert len(emb_db) == len(set(dataloaders["train"].dataset.labels))

    pred = predict(model, dataloaders["test"], emb_db, config.infer.device)
    assert len(pred) == len(dataloaders["test"].dataset)

    db_client.insert_embeddings(emb_db)

    kafka_utils.send_kafka(pred)

    kafka_preds = kafka_utils.get_prediction_from_kafka()
    assert kafka_preds.to_json() == pred.to_json()
