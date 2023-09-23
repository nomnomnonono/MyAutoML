from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


class TableObjective:
    def __init__(
        self,
        x_train,
        y_train,
        model_name,
        main_metric,
        x_valid=None,
        y_valid=None,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.model_name = model_name
        self.main_metric = main_metric

    def __call__(self, trial):
        # classification
        if self.model_name == "LogisticRegression":
            param = {
                "max_iter": 1000,
                "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

        elif self.model_name == "RandomForestClassifier":
            param = {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_depth": trial.suggest_int("max_depth", 1, 1000),
                "max_features": trial.suggest_categorical(
                    "max_features", [None, "sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 1000),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

        elif self.model_name == "LGBMClassifier":
            param = {
                "boosting": "gbdt",
                "objective": "binary"
                if len(self.y_train.unique()) == 2
                else "multiclass",
                "metric": trial.suggest_categorical(
                    "metric",
                    ["binary_logloss", "binary_error"]
                    if len(self.y_train.unique()) == 2
                    else ["multi_logloss", "multi_error"],
                ),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 512),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": trial.suggest_int("seed", 0, 10),
                "verbosity": -1,
            }

            if len(self.y_train.unique()) != 2:
                param["num_class"] = len(self.y_train.unique())

        elif self.model_name == "XGBClassifier":
            param = {
                "verbosity": 0,
                "objective": "binary:logistic"
                if len(self.y_train.unique()) == 2
                else "multi:softmax",
                "tree_method": "exact",
                "booster": trial.suggest_categorical(
                    "booster", ["gbtree", "gblinear", "dart"]
                ),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

            if len(self.y_train.unique()) != 2:
                param["num_class"] = len(self.y_train.unique())

            if param["booster"] in ["gbtree", "dart"]:
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical(
                    "sample_type", ["uniform", "weighted"]
                )
                param["normalize_type"] = trial.suggest_categorical(
                    "normalize_type", ["tree", "forest"]
                )
                param["rate_drop"] = trial.suggest_float(
                    "rate_drop", 1e-8, 1.0, log=True
                )
                param["skip_drop"] = trial.suggest_float(
                    "skip_drop", 1e-8, 1.0, log=True
                )

        # regression
        elif self.model_name == "LinearRegression":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

        elif self.model_name == "RandomForestRegressor":
            param = {
                "criterion": trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_depth": trial.suggest_int("max_depth", 1, 1000),
                "max_features": trial.suggest_categorical(
                    "max_features", [None, "sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 1000),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

        elif self.model_name == "LGBMRegressor":
            param = {
                "boosting": "gbdt",
                "objective": "regression",
                "metric": trial.suggest_categorical("metric", ["l2", "l1", "huber"]),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 512),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": trial.suggest_int("seed", 0, 10),
                "verbosity": -1,
            }

        elif self.model_name == "XGBRegressor":
            param = {
                "verbosity": 0,
                "objective": trial.suggest_categorical(
                    "objective",
                    ["reg:squarederror", "reg:absoluteerror", "reg:squaredlogerror"],
                ),
                "tree_method": "exact",
                "booster": trial.suggest_categorical(
                    "booster", ["gbtree", "gblinear", "dart"]
                ),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

            if param["booster"] in ["gbtree", "dart"]:
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical(
                    "sample_type", ["uniform", "weighted"]
                )
                param["normalize_type"] = trial.suggest_categorical(
                    "normalize_type", ["tree", "forest"]
                )
                param["rate_drop"] = trial.suggest_float(
                    "rate_drop", 1e-8, 1.0, log=True
                )
                param["skip_drop"] = trial.suggest_float(
                    "skip_drop", 1e-8, 1.0, log=True
                )

        else:
            raise ValueError(f"Invalid model: {self.model_name}")

        # create & train model
        model_obj = self.create_model(param)
        model_obj.fit(self.x_train, self.y_train)

        if self.x_valid is None or self.y_valid is None:
            pred = model_obj.predict(self.x_train)
            score = self.main_metric(self.y_train, pred)
        else:
            pred = model_obj.predict(self.x_valid)
            score = self.main_metric(self.y_valid, pred)

        return score

    def create_model(self, param):
        try:
            model = eval(self.model_name)(**param)
        except:
            raise ValueError(f"Invalid model: {self.model_name}")
        return model
