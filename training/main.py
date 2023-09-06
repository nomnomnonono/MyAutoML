import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run(dataset_uri: str, artifact_uri: str) -> None:
    dataset_dir = Path(dataset_uri)
    df_train = pd.read_csv(dataset_dir / "train.csv")
    df_val = pd.read_csv(dataset_dir / "val.csv")
    print(f"Data size: train: {df_train.shape}, val: {df_val.shape}")

    x_train, y_train = df_train["title"], df_train["target"]
    x_val, y_val = df_val["title"], df_val["target"]

    vectorizer = TfidfVectorizer()
    x_train, x_val = vectorizer.fit_transform(x_train), vectorizer.transform(x_val)

    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_train, y_pred)
    print(f"Train accuracy: {acc_train}")

    y_pred = model.predict(x_val)
    acc_val = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc_val}")

    model_dir = Path(artifact_uri)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    print(f"Save model in: {artifact_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--dataset-uri", type=str)
    parser.add_argument("--artifact-uri", type=str)

    args = parser.parse_args()
    run(**vars(args))
