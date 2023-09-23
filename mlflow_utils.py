import json
import os
from io import BytesIO

import mlflow
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv(".env")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")
SERVICE_ACCOUNT_ID = os.environ.get("SERVICE_ACCOUNT_ID")
TRACKING_URI = "http://127.0.0.1:5000/"


def log_metric(expetiment_name, metadata, params):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(expetiment_name.split("-")[0])

    with mlflow.start_run(run_name=expetiment_name):
        for key, value in metadata.items():
            if key == "model_uri":
                mlflow.log_param(key, "gs://" + value.lstrip("/gcs/"))
            else:
                mlflow.log_metric(key, value)

        for key, value in params.items():
            mlflow.log_param(key, value)

    with open(".mlflow_history.txt", "a") as f:
        f.write(expetiment_name + "\n")


def update_mlflow():
    client = storage.Client()
    bucket = client.bucket(ARTIFACT_BUCKET.lstrip("gs://"))

    with open(".mlflow_history.txt", "r") as f:
        history = f.read().split("\n")

    pipeline_blobs = bucket.list_blobs(prefix=SERVICE_ACCOUNT_ID + "/", delimiter="/")
    pipeline_dirs = []
    for page in pipeline_blobs.pages:
        pipeline_dirs.extend(page.prefixes)

    for pipeline_dir in pipeline_dirs:
        blobs = bucket.list_blobs(prefix=pipeline_dir + "train")
        metadata, params = None, None
        for blob in blobs:
            if os.path.split(blob.name)[-1] == "executor_output.json":
                content = blob.download_as_bytes()
                result = json.load(BytesIO(content))
                metadata = result["artifacts"]["metrics"]["artifacts"][0]["metadata"]
                experiment_name = blob.name.split("/")[1]
            if os.path.split(blob.name)[-1] == "best_params.json":
                content = blob.download_as_bytes()
                result = json.load(BytesIO(content))
                params = result

        if (
            metadata is not None
            and params is not None
            and experiment_name not in history
        ):
            log_metric(experiment_name, metadata, params)


if __name__ == "__main__":
    update_mlflow()
