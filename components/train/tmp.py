import os

from dotenv import load_dotenv
from kfp.v2.dsl import InputPath, Metrics, Output, OutputPath, component

load_dotenv(".env")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
AR_REPOSITORY_NAME = os.environ.get("AR_REPOSITORY_NAME")


@component(
    base_image=f"asia-northeast1-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/train:latest"
)
def train(
    dataset_uri: InputPath("Dataset"),
    artifact_uri: OutputPath("Model"),
    metrics: Output[Metrics],
) -> None:
    from pathlib import Path

    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

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

    metrics.log_metric("train accuracy", acc_train)
    metrics.log_metric("validation accuracy", acc_val)
    metrics.log_metric("model uri", artifact_uri)
