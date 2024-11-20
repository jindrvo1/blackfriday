FROM python:3.11

WORKDIR /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install kfp google-cloud-aiplatform google-cloud-secret-manager