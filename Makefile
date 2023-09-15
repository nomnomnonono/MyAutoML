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
	poetry run pip install google-cloud-aiplatform
	poetry run pip install streamlit
	poetry run pip install altair==4.2.2

.PHONY: streamlit
streamlit:
	poetry run streamlit run app.py

.PHONY: mlflow
mlflow:
	poetry run mlflow ui
