from pathlib import Path

import cv2
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import read_config
from src.logger import LOGGER


def resize_img(img_p) -> None:
    img_p = str(img_p)
    img = cv2.imread(img_p)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(img_p, img)


def resize_images(data_dir: Path) -> None:
    _ = Parallel(n_jobs=-1)(
        delayed(resize_img)(img_path)
        for img_path in tqdm(list(data_dir.rglob("*.jpg")))
    )


def split_data(data: pd.DataFrame) -> pd.DataFrame:
    # get 400 classes with 2 images + 100 new_whale
    data["stage"] = "train"
    new_whale_idxs = data.loc[data["Id"] == "new_whale"].index.tolist()
    num_val_classes = 400
    for id_, cnt in data["Id"].value_counts().items():
        if num_val_classes == 0:
            break
        if cnt == 2:
            data.loc[data["Id"] == id_, "stage"] = "val"
            data.loc[data.index.isin(new_whale_idxs[:100]), "stage"] = "val"
            new_whale_idxs = new_whale_idxs[100:]
            num_val_classes -= 1
    return data


def preprocess(data_dir: str) -> bool:
    data_dir = Path(data_dir)
    LOGGER.info("Resizing images")
    resize_images(data_dir)
    data = pd.read_csv(data_dir / "train.csv")
    LOGGER.info("Split data")
    data = split_data(data)
    data.to_csv(data_dir / "metadata.csv", index=False)
    return (data_dir / "metadata.csv").exists()


if __name__ == "__main__":
    cfg_path = Path(__file__).parent / "config.yml"
    cfg = read_config(cfg_path)
    preprocess(cfg.data.data_dir)
