.PHONY: format
format:
	poetry run pysen run format

.PHONY: lint
lint:
	poetry run pysen run lint

.PHONY: install
install:
	poetry install
	poetry run pip install --upgrade pip
	poetry run pip install kfp
	poetry run pip install google-cloud-aiplatform
	poetry run pip install protobuf==3.20

.PHONY: pipeline
pipeline:
	poetry run python pipeline.py

.PHONY: mlflow
mlflow:
	poetry run mlflow ui
