# see: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
from pathlib import Path

import pytest
import pandas as pd

from src.utils import read_config
from src.net import Net
from src.dataset import get_dataloaders

DEFAULT_CFG_PATH = Path(__file__).parent / "../../config.yml"
DEFAULT_TINY_DATA_DIR = Path(__file__).parent / "../../tests/tiny_dataset"


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--skip-integration", action="store_true", default=False, help="skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(
            reason="you need to remove --skip-slow option to run"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if config.getoption("--skip-integration"):
        skip_slow = pytest.mark.skip(
            reason="you need to remove --skip-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture()
def config():
    cfg = read_config(DEFAULT_CFG_PATH)
    cfg.data.data_dir = "tests/tiny_dataset"
    cfg.train.net.num_classes = 2
    cfg.train.loss.num_classes = 2
    cfg.train.batch_size = 1
    cfg.infer.batch_size = 1
    cfg.train.device = "cpu"
    return cfg


@pytest.fixture()
def model(config):
    net = Net(config.train)
    net.to(config.infer.device)
    net.eval()
    return net


@pytest.fixture()
def tyny_dataset():
    return pd.read_csv(DEFAULT_TINY_DATA_DIR / "train.csv")


@pytest.mark.slow
@pytest.fixture()
def dataloaders(config):
    train_dl, val_dl, test_dl = get_dataloaders(
        Path(config.data.data_dir), config.infer.batch_size
    )
    return {"train": train_dl, "val": val_dl, "test": test_dl}
