from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
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
                "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "RandomForestClassifier":
            param = {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", ["True", "False"]),
                "max_depth": trial.suggest_int("max_depth", 1, 1000),
                "max_features": trial.suggest_categorical(
                    "max_features", ["None", "sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 1000),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "LGBMClassifier":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "XGBClassifier":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }

        # regression
        elif self.model_name == "LinearRegression":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "RandomForestRegressor":
            param = {
                "criterion": trial.suggest_categorical(
                    "criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", ["True", "False"]),
                "max_depth": trial.suggest_int("max_depth", 1, 1000),
                "max_features": trial.suggest_categorical(
                    "max_features", ["None", "sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 1, 1000),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "LGBMRegressor":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
        elif self.model_name == "XGBRegressor":
            param = {
                "random_state": trial.suggest_int("random_state", 0, 10),
            }
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
