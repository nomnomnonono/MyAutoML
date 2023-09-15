FROM python:3.10-slim
WORKDIR /root

RUN apt-get update && apt-get install -y libgomp1

RUN pip install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install
RUN pip install kfp

COPY . .

ENTRYPOINT ["python", "main.py"]
