import os

from dotenv import load_dotenv
from kfp.v2.dsl import OutputPath, component

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")


@component(
    base_image=f"asia-northeast1-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/preprocess:latest"
)
def preprocess(
    data_path: str, data_type: str, target_task: str, dataset_uri: OutputPath("Dataset")
) -> None:
    import glob
    import os
    from pathlib import Path

    import pandas as pd

    if data_type == "table":
        dataset_dir = Path(dataset_uri)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "valid", "test"]:
            df = pd.read_csv(
                glob.glob(
                    os.path.join("/gcs", data_path.lstrip("gs://"), split, "*.csv")
                )[0]
            )
            print(
                f"Load CSV from: {glob.glob(os.path.join('/gcs', data_path.lstrip('gs://'), split, '*.csv'))[0]}"
            )

            # preprocess

            split_dir = dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(split_dir / "data.csv", index=False)
            print(f"Save train/val data in: {dataset_dir}")
    elif data_type == "text":
        pass
    elif data_type == "image":
        pass
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
