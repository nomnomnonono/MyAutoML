FROM python:3.10-slim
WORKDIR /root

RUN pip install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install
RUN pip install kfp

COPY . .

ENTRYPOINT ["python", "main.py"]
