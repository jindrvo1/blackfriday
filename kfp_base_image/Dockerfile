FROM python:3.11

WORKDIR /app

# Create a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install common packages
RUN pip install --upgrade pip
RUN pip install \
            kfp \
            google-cloud-aiplatform \
            google-cloud-secret-manager \
            cloudml-hypertune \
            python-dotenv

# Install GCP CLI
RUN curl -o gcp-cli.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
RUN tar -xf gcp-cli.tar.gz
RUN ./google-cloud-sdk/install.sh
ENV PATH="$PATH:/app/google-cloud-sdk/bin"
RUN rm gcp-cli.tar.gz

# Configure GCP
COPY gcs_sa.json /gcs_sa.json

ENV GOOGLE_APPLICATION_CREDENTIALS="/gcs_sa.json"
ENV GOOGLE_CLOUD_PROJECT='ml-spec-demo2'
ENV GOOGLE_CLOUD_REGION='europe-west3'

RUN gcloud auth activate-service-account --key-file /gcs_sa.json
RUN gcloud config set project ml-spec-demo2
RUN gcloud config set compute/zone europe-west3
RUN gcloud config set compute/region europe-west3
RUN gcloud config set ai/region europe-west3
