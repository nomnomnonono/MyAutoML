# MyAutoML
## Requirements
- Poetry
- gcloud CLI
- docker compose

## Setup
### GCP Authentification
```bash
$ gcloud auth login
$ gcloud components install pubsub-emulator
```

### Install Dependencies
```bash
$ make install
```

### Environmental Variables
```bash
$ vi .env
```

- 以下の情報を記入＋環境変数としてexportしておく
```bash
GCP_PROJECT_ID=your project id
LOCATION=asia-northeast1
AR_REPOSITORY_NAME=artifact registory repository name
DATA_BUCKET=gs://xxx
ARTIFACT_BUCKET=gs://yyy
```

### Create Cloud Storage Bucket
```bash
$ gsutil mb -l $LOCATION $DATA_BUCKET
$ gsutil mb -l $LOCATION $ARTIFACT_BUCKET
```

## Build & Push Docker Image
```bash
$ gcloud auth configure-docker asia-northeast1-docker.pkg.dev
$ gcloud artifacts repositories create $AR_REPOSITORY_NAME --location=$LOCATION --repository-format=docker
$ docker compose build
$ docker compose push
```

## Boot Streamlit GUI
```bash
$ make streamlit
```

## Boot MLflow Server
```bash
$ make mlflow
```
