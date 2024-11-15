FROM --platform=linux/amd64 python:3.11
# FROM python:3.11

WORKDIR /app

COPY run_pipeline.py /app/run_pipeline.py
COPY blackfriday_pipeline.yaml /app/blackfriday_pipeline.yaml
COPY gcs_sa.json /app/gcs_sa.json

RUN pip install google-cloud-aiplatform kfp

ENTRYPOINT ["python", "run_pipeline.py"]