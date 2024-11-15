region = 'europe-west3'
project_id = 'ml-spec-demo2'
service_account = 'gcs-sa@ml-spec-demo2.iam.gserviceaccount.com'

bucket = 'gs://blackfridaydataset'
pipeline_root = f'{bucket}/pipeline_root'
package_path = 'blackfriday_pipeline.yaml'

data_folder = 'source_data'
train_file = 'train.csv'
test_file = 'test.csv'

gcs_train_data_path = f'{bucket}/{data_folder}/{train_file}'
gcs_test_data_path = f'{bucket}/{data_folder}/{test_file}'

pipeline_kwargs = {
    'gcs_train_data_path': gcs_train_data_path,
    'gcs_test_data_path': gcs_test_data_path,
    'project_id': project_id,
    'region': region,
    'service_account': service_account,
    'pipeline_root': pipeline_root,
    'package_path': 'blackfriday_pipeline.yaml',
}


def run_custom_job(credentials):
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "ACCELERATOR_TYPE_UNSPECIFIED",
                "accelerator_count": 0,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": 'europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/blackfriday-pipeline:latest',
                "command": [],
                "args": [],
            }
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name=package_path.split('.')[0],
        worker_pool_specs=worker_pool_specs,
    )

    custom_job.run()


if __name__ == '__main__':
    import json

    from google.cloud import aiplatform
    from google.oauth2 import service_account

    with open('gcs_sa.json', 'r') as f:
        sa_key = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(sa_key)

    aiplatform.init(
        project=project_id,
        location=region,
        # service_account=service_account,
        credentials=credentials,
        staging_bucket=f'gs://blackfriday_staging'
    )

    run_custom_job(credentials)
