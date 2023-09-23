import json
from pathlib import Path

import joblib
import optuna
from kfp.v2.dsl import InputPath, Metrics, Output, OutputPath
from objective import TableObjective
from table_utils import get_table_dataset, get_table_metric_dict, log_table_metrics


def train_table(
    dataset_uri: InputPath("Dataset"),
    target_task: str,
    model_name: str,
    main_metric: str,
    params: str,
    artifact_uri: OutputPath("Model"),
    metrics: Output[Metrics],
) -> None:
    params = json.loads(params.replace("'", '"'))
    for key, values in params.items():
        params[key] = [None if value == "None" else value for value in values]

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_table_dataset(dataset_uri)
    metric_list = get_table_metric_dict(target_task)
    if len(metric_list[main_metric]) == 2:
        main_metric, is_high = metric_list[main_metric]
    elif len(metric_list[main_metric]) == 3:
        main_metric, is_high, metric_param = metric_list[main_metric]
    else:
        raise ValueError(f"Invalid metric list: {main_metric}")

    objective = TableObjective(
        x_train=x_train,
        y_train=y_train,
        model_name=model_name,
        main_metric=main_metric,
        x_valid=x_valid,
        y_valid=y_valid,
    )

    study = optuna.create_study(direction="maximize" if is_high else "minimize")
    study.optimize(objective, n_trials=100)

    # log best metrics
    best_model = objective.create_model(study.best_params)
    best_model.fit(x_train, y_train)

    for split, data in {
        "train": (x_train, y_train),
        "valid": (x_valid, y_valid),
        "test": (x_test, y_test),
    }.items():
        pred = best_model.predict(data[0])
        log_table_metrics(
            split,
            data[1],
            pred,
            metric_list,
            metrics,
            is_multi=len(y_train.unique()) != 2,
        )

    # log artifact uri
    metrics.log_metric("model_uri", artifact_uri)

    # save model
    model_dir = Path(artifact_uri)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "model.joblib")

    # save parameters
    with open(model_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(model_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
