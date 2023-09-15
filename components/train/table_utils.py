from pathlib import Path

import numpy as np
import pandas as pd
from kfp.v2.dsl import Metrics, Output
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def get_table_metric_dict(target_task):
    if target_task == "classification":
        return {
            "Accuracy": (accuracy_score, True),
            "Precision": (precision_score, True),
            "Recall": (recall_score, True),
            "F1": (f1_score, True),
            "AUC": (roc_auc_score, True),
        }
    elif target_task == "regression":
        return {
            "MSE": (mean_squared_error, False),
            "RMSE": (mean_squared_error, False, {"squared": False}),
            "MAE": (mean_absolute_error, False),
            "R2": (r2_score, True),
        }
    else:
        return []


def get_table_dataset(dataset_uri: str):
    dataset_dir = Path(dataset_uri)
    df_train = pd.read_csv(dataset_dir / "train" / "data.csv")
    df_val = pd.read_csv(dataset_dir / "valid" / "data.csv")
    df_test = pd.read_csv(dataset_dir / "test" / "data.csv")
    return (
        df_train.drop("target", axis=1),
        df_train["target"],
        df_val.drop("target", axis=1),
        df_val["target"],
        df_test.drop("target", axis=1),
        df_test["target"],
    )


def log_table_metrics(
    split: str,
    label: np.ndarray,
    pred: np.ndarray,
    metric_dict: dict[str, tuple],
    metrics: Output[Metrics],
):
    for metric_name, metric_value in metric_dict.keys():
        score = metric_value[0](label, pred)
        metrics.log_metric(f"{split}/{metric_name.lower()}", score)
