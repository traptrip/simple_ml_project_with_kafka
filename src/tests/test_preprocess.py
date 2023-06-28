from preprocess import preprocess, split_data
from src.tests.conftest import DEFAULT_TINY_DATA_DIR


def test_split_data(tyny_dataset):
    processed_data = split_data(tyny_dataset)
    assert (
        "stage" in processed_data.columns
        and processed_data.loc[processed_data.stage == "val"].shape[0] > 0
    )


def test_save_splitted_data():
    assert preprocess(DEFAULT_TINY_DATA_DIR)
