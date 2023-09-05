import os
from enum import Enum

import joblib
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI")
AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

app = FastAPI()


def build_model(artifact_uri: str) -> tuple[LogisticRegression, TfidfVectorizer]:
    bucket_name, model_dir = artifact_uri.lstrip("gs://").split("/", maxsplit=1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    model_blob = bucket.blob(f"{model_dir}/model.joblib")
    model_blob.download_to_filename("model.joblib")
    vectorizer_blob = bucket.blob(f"{model_dir}/vectorizer.joblib")
    vectorizer_blob.download_to_filename("vectorizer.joblib")
    return joblib.load("model.joblib"), joblib.load("vectorizer.joblib")


model, vectorizer = build_model(AIP_STORAGE_URI)


class Category(Enum):
    CV = 0
    CL = 1
    RO = 2


class PaperTitle(BaseModel):
    title: str


class Prediction(BaseModel):
    category: str


class Predictions(BaseModel):
    predictions: list[Prediction]


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {"health": "ok"}


@app.post(
    AIP_PREDICT_ROUTE, response_model=Predictions, response_model_exclude_unset=True
)
async def predict(instances: list[PaperTitle]):
    instances = [x["title"] for x in instances]
    instances = vectorizer.transform(instances)
    preds = model.predict(instances)

    outputs = []
    for pred in preds:
        specie = Category(pred).name
        outputs.append(Prediction(specie="cs." + specie))

    return Predictions(predictions=outputs)
