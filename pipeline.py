import argparse
import datetime
import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from kfp import compiler, dsl

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")
LOCATION = os.environ.get("LOCATION")
DATA_BUCKET = os.environ.get("DATA_BUCKET")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME")

from components.preprocess.main import preprocess
from components.train.main import train


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="MyAutoML",
    pipeline_root=ARTIFACT_BUCKET,
)
def pipeline(
    dataset: str,
    data_type: str,
    target_task: str,
    model_name: str,
    main_metric: str,
    machine_type: str,
    is_train: bool,
) -> None:
    if is_train:
        preprocess_op = preprocess(
            data_path=dataset,
            data_type=data_type,
            target_task=target_task,
        )

        train_op = train(
            dataset_uri=preprocess_op.output,
            data_type=data_type,
            target_task=target_task,
            model_name=model_name,
            main_metric=main_metric,
        )
        train_op.custom_job_spec = {
            "jobSpec": {
                "workerPoolSpecs": [
                    {
                        "machineSpec": {"machineType": machine_type},
                        "replicaCount": 1,
                    }
                ],
            },
        }

    else:
        pass
        """
        deploy_op = components.load_component_from_file("components/deploy/component.yaml")
        _ = deploy_op(
            artifact=train_task.outputs["artifact"],
            model_name="ml-pipeline-arxiv-paper-model",
            serving_container_image_uri=f"asia-northeast1-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/serving:latest",
            serving_container_environment_variables='{"APP_MODULE": "server:app"}',
            serving_container_ports=80,
            endpoint_name="ml-pipeline-arxiv-paper-endpoint",
            deploy_name="ml-pipeline-arxiv-paper-deploy",
            machine_type="n1-standard-2",
            min_replicas=1,
            project=PROJECT_ID,
            location=LOCATION,
        )
        """


def exec_pipeline(
    dataset,
    data_type,
    target_task,
    model_name,
    main_metric,
    machine_type,
    is_train=True,
):
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=f"{dataset}-{model_name}.json".lower(),
    )

    job = aiplatform.PipelineJob(
        display_name=f"{dataset}-{model_name}".lower(),
        template_path=f"{dataset}-{model_name}.json".lower(),
        job_id=f"{dataset}-{model_name}".lower()
        + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-4]}",
        pipeline_root=ARTIFACT_BUCKET,
        enable_caching=False,
        project=PROJECT_ID,
        location=LOCATION,
        parameter_values={
            "dataset": os.path.join(DATA_BUCKET, dataset),
            "data_type": data_type,
            "target_task": target_task,
            "model_name": model_name,
            "main_metric": main_metric,
            "machine_type": machine_type,
            "is_train": is_train,
        },
    )

    job.submit()
    print("Pipelien execution success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exec pipeline")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--target_task", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--main_metric", type=str)
    parser.add_argument("--machine_type", type=str)
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()

    exec_pipeline(**vars(args))
