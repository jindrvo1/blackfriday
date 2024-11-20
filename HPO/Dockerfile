FROM europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/kfp_base_image:latest

WORKDIR /app

COPY .env /app/.env
COPY run_pipeline.py /app/run_pipeline.py
COPY blackfriday_pipeline.yaml /app/blackfriday_pipeline.yaml
COPY secrets_manager_sa.json /app/secrets_manager_sa.json

ENTRYPOINT ["python", "run_pipeline.py"]