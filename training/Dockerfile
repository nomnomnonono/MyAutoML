FROM python:3.10-slim
WORKDIR /root

RUN pip install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install

COPY . .

ENTRYPOINT ["python", "main.py"]
