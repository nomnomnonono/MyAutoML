import itertools
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import xgboost as xgb
from kfp.v2.dsl import InputPath, Metrics, Output, OutputPath
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from table_utils import get_table_dataset, get_table_metric_dict, log_table_metrics

SEEDS = [42]


def train_table(
    dataset_uri: InputPath("Dataset"),
    target_task: str,
    model_name: str,
    main_metric: str,
    artifact_uri: OutputPath("Model"),
    metrics: Output[Metrics],
) -> None:
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_table_dataset(dataset_uri)
    metric_list = get_table_metric_dict(target_task)
    if len(metric_list[main_metric]) == 2:
        main_metric, is_high = metric_list[main_metric]
    elif len(metric_list[main_metric]) == 3:
        main_metric, is_high, metric_param = metric_list[main_metric]
    else:
        raise ValueError(f"Invalid metric list: {main_metric}")

    if target_task == "classification":
        if model_name == "LogisticRegression":
            model_obj = LogisticRegression
            params = {
                "penalty": ["l1", "l2"],
                "C": [0.1, 1.0, 10],
                "random_state": SEEDS,
            }
        elif model_name == "RandomForestClassifier":
            model_obj = RandomForestClassifier
            params = {
                "n_estimators": [10, 100],
                "max_depth": [3, 5],
                "random_state": SEEDS,
            }
        elif model_name == "LightGBMClassifier":
            model_obj = lgb.LGBMClassifier
            params = {
                "objective": "classification",
                "random_state": SEEDS,
            }
        elif model_name == "XGBoostClassifier":
            model_obj = xgb.XGBClassifier
            params = {
                "objective": "binary:logistic",
                "random_state": SEEDS,
            }
        else:
            raise ValueError(f"Invalid model: {model_name}")

        # parameter search phase
        keys = list(params.keys())
        model, best_score, best_params = None, 0, {}
        for param in list(itertools.product(*list(params.values()))):
            _model = model_obj(**dict(zip(keys, param)))
            _model.fit(x_train, y_train)
            pred = _model.predict(x_valid)
            score = main_metric(y_valid, pred)

            # update best model according to main metric
            if is_high and score > best_score:
                best_score = score
                model = _model
                best_params = dict(zip(keys, param))
            elif not is_high and score < best_score:
                best_score = score
                model = _model
                best_params = dict(zip(keys, param))

        # log metrics for all data splits
        model = model_obj(**best_params)
        model.fit(x_train, y_train)
        for split, data in {
            "train": (x_train, y_train),
            "valid": (x_valid, y_valid),
            "test": (x_test, y_test),
        }:
            pred = model.predict(data[0])
            log_table_metrics(split, data[1], pred, metric_list, metrics)

    elif target_task == "regression":
        if model_name == "LinearRegression":
            model = LinearRegression
            params = {
                "penalty": ["l1", "l2"],
                "C": [0.1, 1.0, 10],
                "random_state": SEEDS,
            }
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor
            params = {
                "n_estimators": [10, 100],
                "max_depth": [3, 5],
                "random_state": SEEDS,
            }
        elif model_name == "LightGBMRegressor":
            model = lgb.LGBMRegressor
            params = {
                "objective": "regression",
                "random_state": SEEDS,
            }
        elif model_name == "XGBoostRegressor":
            model = xgb.XGBRegressor
            params = {
                "objective": "reg:squarederror",
                "random_state": SEEDS,
            }
        else:
            raise ValueError(f"Invalid model: {model_name}")

        # parameter search phase
        keys = list(params.keys())
        model, best_score, best_params = None, 0, {}
        for param in list(itertools.product(*list(params.values()))):
            _model = model_obj(**dict(zip(keys, param)))
            _model.fit(x_train, y_train)
            pred = _model.predict(x_valid)
            score = main_metric(y_valid, pred)

            # update best model according to main metric
            if is_high and score > best_score:
                best_score = score
                model = _model
                best_params = dict(zip(keys, param))
            elif not is_high and score < best_score:
                best_score = score
                model = _model
                best_params = dict(zip(keys, param))

        # log metrics for all data splits
        model = model_obj(**best_params)
        model.fit(x_train, y_train)
        for split, data in {
            "train": (x_train, y_train),
            "valid": (x_valid, y_valid),
            "test": (x_test, y_test),
        }:
            pred = model.predict(data[0])
            log_table_metrics(split, data[1], pred, metric_list, metrics)

    else:
        raise ValueError(f"Invalid target_task: {target_task}")

    # log artifact uri
    metrics.log_metric("model_uri", artifact_uri)

    # save model
    model_dir = Path(artifact_uri)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    # save best parameters
    with open(model_dir / "params.json", "w") as f:
        json.dump(best_params, f, indent=4)
