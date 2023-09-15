FROM python:3.10-slim
WORKDIR /root

# Upgrade pip to latest
RUN pip install -U pip \
    && pip install -U google-cloud-aiplatform google-cloud-storage fsspec gcsfs

COPY main.py ./main.py

ENTRYPOINT ["python", "main.py"]
