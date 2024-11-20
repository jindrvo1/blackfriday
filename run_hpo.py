from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


region = 'europe-west3'
project_id = 'ml-spec-demo2'

bucket = 'gs://blackfridaydataset'
pipeline_root = f'{bucket}/pipeline_root'
package_name = 'blackfriday_pipeline'
package_path = f'blackfriday_pipeline.yaml'

staging_bucket = f'gs://blackfriday_staging'

data_folder = 'source_data'
train_file = 'train.csv'
test_file = 'test.csv'

gcs_train_data_path = f'{bucket}/{data_folder}/{train_file}'
gcs_test_data_path = f'{bucket}/{data_folder}/{test_file}'

docker_image_uri = 'europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/blackfriday-pipeline:latest'

worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-4",
            "accelerator_type": None,
            "accelerator_count": None,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": docker_image_uri,
            "command": [],
            "args": [],
        }
    }
]


def create_custom_job(package_name, worker_pool_specs) -> aiplatform.CustomJob:
    custom_job = aiplatform.CustomJob(
        display_name=package_name,
        worker_pool_specs=worker_pool_specs,
    )

    return custom_job


def run_hpo(parameters, metric, package_name, worker_pool_specs):
    custom_job = create_custom_job(package_name, worker_pool_specs)

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=f"hpo_{package_name}_min_val_rmse",
        custom_job=custom_job,
        metric_spec=metric,
        parameter_spec=parameters,
        max_trial_count=2,
        parallel_trial_count=1,
    )

    hpt_job.run()


if __name__ == '__main__':
    parameters = {
        'n_estimators': hpt.IntegerParameterSpec(min=100, max=500, scale='linear'),
        'max_depth': hpt.IntegerParameterSpec(min=3, max=30, scale='linear'),
        'min_child_weight': hpt.IntegerParameterSpec(min=1, max=10, scale='linear'),
        'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='linear'),
    }

    metric = {'val_rmse': 'minimize'}

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket
    )

    run_hpo(parameters, metric, package_name, worker_pool_specs)