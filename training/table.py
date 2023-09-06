import glob
import itertools
import os

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_dataset(data_fuse_path):
    train = pd.read_csv(glob.glob(os.path.join(data_fuse_path, "train", "*.csv"))[0])
    valid = pd.read_csv(glob.glob(os.path.join(data_fuse_path, "valid", "*.csv"))[0])
    test = pd.read_csv(glob.glob(os.path.join(data_fuse_path, "test", "*.csv"))[0])
    y_train, y_valid, y_test = train["target"], valid["target"], test["target"]
    x_train, x_valid, x_test = train.drop("target", axis=1), valid.drop("target", axis=1), test.drop("target", axis=1)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def train_table(data_fuse_path, model_fuse_path, target_task, model, main_metric, sub_metric):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataset(data_fuse_path)

    if target_task == "classification":
        if model == "LogisticRegression":
            model = LogisticRegression
            params = None
        elif model == "RandomForestClassifier":
            model = RandomForestClassifier
            params = {"n_estimators": [10, 100], "max_depth": [3, 5]}
        elif model == "LightGBMClassifier":
            model = lgb.LGBMClassifier
            params = {"objective": "classification"}
        elif model == "XGBoostClassifier":
            model = xgb.XGBClassifier
            params = {"objective": "binary:logistic"}
        else:
            raise ValueError(f"Invalid model: {model}")
        
        if params is None:
            model.fit(x_train, y_train)
        else:
            keys = list(params.keys())
            for param in list(itertools.product(*list(params.values()))):
                _model = model(**dict(zip(keys, param)))
                _model.fit(x_train, y_train)
    elif target_task == "regression":
        if model == "LinearRegression":
            model = LinearRegression
            params = None
        elif model == "RandomForestRegressor":
            model = RandomForestRegressor
            params = {"n_estimators": [10, 100], "max_depth": [3, 5]}
        elif model == "LightGBMRegressor":
            model = lgb.LGBMRegressor
            params = {"objective": "regression"}
        elif model == "XGBoostRegressor":
            model = xgb.XGBRegressor
            params = {"objective": "reg:squarederror"}
        else:
            raise ValueError(f"Invalid model: {model}")
        
        if params is None:
            model.fit(x_train, y_train)
        else:
            keys = list(params.keys())
            for param in list(itertools.product(*list(params.values()))):
                _model = model(**dict(zip(keys, param)))
                _model.fit(x_train, y_train)
    else:
        raise ValueError(f"Invalid target_task: {target_task}")
