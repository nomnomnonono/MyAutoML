import argparse
import os
import re

from table import train_table


def get_fuse_path(path: str) -> str:
    pattern = "gs://(?P<bucket>[^/]+)/(?P<key>.+)"
    match = re.match(pattern, path)
    return f"/gcs/{match.group('bucket')}/{match.group('key')}"


def run(dataset: str, data_type: str, target_task: str, model: str, main_metric: str, sub_metric: list[str]) -> None:
    data_fuse_path = get_fuse_path(dataset)
    model_fuse_path = get_fuse_path(os.getenv("AIP_MODEL_DIR"))

    if data_type == "tabel":
        train_table(data_fuse_path, model_fuse_path, target_task, model, main_metric, sub_metric)
    elif data_type == "text":
        pass
    elif data_type == "image":
        pass
    else:
        raise ValueError(f"Invalid data_type: {data_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--target_task", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--main_metric", type=str)
    parser.add_argument("--sub_metric", type=list[str])
    args = parser.parse_args()
    run(**vars(args))
