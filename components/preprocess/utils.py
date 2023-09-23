import glob
import os

import pandas as pd


def get_table_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        glob.glob(os.path.join("/gcs", data_path.lstrip("gs://"), "train", "*.csv"))[0]
    )
    valid = pd.read_csv(
        glob.glob(os.path.join("/gcs", data_path.lstrip("gs://"), "valid", "*.csv"))[0]
    )
    test = pd.read_csv(
        glob.glob(os.path.join("/gcs", data_path.lstrip("gs://"), "test", "*.csv"))[0]
    )
    return train, valid, test
