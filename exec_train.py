import os

from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")
DATA_BUCKET = os.environ.get("DATA_BUCKET")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")


def exec_train_job(dataset, data_type, target_task, model, main_metric, sub_metric, machine_type):
    custom_job = aiplatform.CustomContainerTrainingJob(
        display_name=f"{dataset}-{model}".lower(),
        container_uri=f"gcr.io/{PROJECT_ID}/{AR_REPOSITORY_NAME}/train:latest",
        staging_bucket=ARTIFACT_BUCKET
    )

    custom_job.run(
        machine_type=machine_type,
        model_display_name=f"{dataset}-{model}".lower(),
        args=[
            f"--dataset={os.path.join(DATA_BUCKET, dataset)}",
            f"--data_type={data_type}",
            f"--target_task={target_task}",
            f"--model={model}",
            f"--main_metric={main_metric}",
            f"--sub_metric={','.join(sub_metric)}",
        ],
    )
