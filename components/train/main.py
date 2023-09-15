import os

from dotenv import load_dotenv
from kfp.v2.dsl import InputPath, Metrics, Output, OutputPath, component

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")


@component(
    base_image=f"asia-northeast1-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/train:latest"
)
def train(
    data_type: str,
    target_task: str,
    model_name: str,
    main_metric: str,
    dataset_uri: InputPath("Dataset"),
    artifact_uri: OutputPath("Model"),
    metrics: Output[Metrics],
) -> None:
    if data_type == "tabel":
        from components.train.table import train_table

        train_table(
            dataset_uri,
            artifact_uri,
            metrics,
            target_task,
            model_name,
            main_metric,
        )
    elif data_type == "text":
        pass
    elif data_type == "image":
        pass
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
