docker build -f Dockerfile --platform linux/amd64 -t europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/blackfriday-pipeline:latest .
docker push europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/blackfriday-pipeline:latest
