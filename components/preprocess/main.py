import os

from dotenv import load_dotenv
from kfp.v2.dsl import Dataset, OutputPath, component

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")


@component(
    base_image=f"asia-northeast1-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/preprocess:latest"
)
def preprocess(
    data_path: str, data_type: str, dataset_uri: OutputPath(Dataset)
) -> None:
    from pathlib import Path

    import pandas as pd
    from utils import get_table_data

    if data_type == "table":
        train, valid, test = get_table_data(data_path)
        data = pd.concat([train, valid, test], axis=0)

        for column in data.columns:
            # convert string to int
            if data[column].dtype == "object":
                dic = {v: i for i, v in enumerate(list(data[column].unique()))}
                data[column] = data[column].astype("category")
                data[column] = data[column].map(dic)

            # fill na with median / mode
            if data[column].isnull().sum() > 0:
                if len(data[column].unique()) < 100:
                    data[column].fillna(data[column].mode()[0], inplace=True)
                else:
                    data[column].fillna(data[column].median(), inplace=True)

        dataset_dir = Path(dataset_uri)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        for name, length in {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        }.items():
            split_dir = dataset_dir / name
            split_dir.mkdir(parents=True, exist_ok=True)

            df = data.iloc[idx : idx + length]
            df.to_csv(split_dir / "data.csv", index=False)
            idx += length

    elif data_type == "text":
        pass
    elif data_type == "image":
        pass
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
