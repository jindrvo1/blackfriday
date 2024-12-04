import argparse
from dotenv import dotenv_values
import json

from google.cloud import aiplatform, secretmanager
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.oauth2 import service_account


env_vars = dotenv_values('.env')

PROJECT_ID = env_vars['PROJECT_ID']
REGION = env_vars['REGION']
SERVICE_ACCOUNT = env_vars['SERVICE_ACCOUNT']
SA_SECRETS_NAME = env_vars['SA_SECRETS_NAME']
SECRETS_SERVICE_ACCOUNT = env_vars['SECRETS_SERVICE_ACCOUNT']
GCS_TRAIN_DATA_PATH = env_vars['GCS_TRAIN_DATA_PATH']
GCS_TEST_DATA_PATH = env_vars['GCS_TEST_DATA_PATH']
PIPELINE_ROOT = env_vars['PIPELINE_ROOT']
PREPARE_DATA_PACKAGE_PATH = env_vars['PREPARE_DATA_PACKAGE_PATH']
TRAIN_MODEL_PACKAGE_PATH = env_vars['TRAIN_MODEL_PACKAGE_PATH']
KFP_BASE_IMAGE = env_vars['KFP_BASE_IMAGE']
TRAIN_MODEL_COMPONENT_IMAGE = env_vars['TRAIN_MODEL_COMPONENT_IMAGE']
GCS_STAGING_BUCKET = env_vars['GCS_STAGING_BUCKET']


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcs_train_data_path', type=str, default=GCS_TRAIN_DATA_PATH)
    parser.add_argument('--gcs_test_data_path', type=str, default=GCS_TEST_DATA_PATH)

    parser.add_argument('--project_id', type=str, default=PROJECT_ID)
    parser.add_argument('--region', type=str, default=REGION)
    parser.add_argument('--service_account', type=str, default=SERVICE_ACCOUNT)
    parser.add_argument('--sa_secrets_name', type=str, default=SA_SECRETS_NAME)
    parser.add_argument('--secrets_service_account', type=str, default=SECRETS_SERVICE_ACCOUNT)
    parser.add_argument('--pipeline_root', type=str, default=PIPELINE_ROOT)
    parser.add_argument('--prepare_data_package_path', type=str, default=PREPARE_DATA_PACKAGE_PATH)
    parser.add_argument('--train_model_package_path', type=str, default=TRAIN_MODEL_PACKAGE_PATH)
    parser.add_argument('--gcs_staging_bucket', type=str, default=GCS_STAGING_BUCKET)

    return parser


def get_credentials(secrets_service_account: str, secret_name: str) -> service_account.Credentials:
    with open(secrets_service_account, 'r') as f:
        secret_manager_sa_key = json.load(f)
        credentials_secret_manager = service_account.Credentials.from_service_account_info(secret_manager_sa_key)

        secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=credentials_secret_manager)
        response = secret_manager_client.access_secret_version(name=secret_name)
        service_account_info = response.payload.data.decode("UTF-8")

    credentials = service_account.Credentials.from_service_account_info(json.loads(service_account_info))

    return credentials


def init_aiplatform(args: argparse.Namespace) -> None:
    # sa_secrets_name = f'projects/{args.project_id}/secrets/{args.sa_secrets_name}/versions/latest'
    # credentials = get_credentials(args.secrets_service_account, sa_secrets_name)

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        # credentials=credentials,
    )


def run_prepare_data_pipeline(
    service_account: str,
    pipeline_root: str,
    package_path: str,
) -> dict[str, str]:
    job = aiplatform.PipelineJob(
        display_name=package_path.split('.')[0],
        template_path=package_path,
        pipeline_root=pipeline_root,
        enable_caching=True,
    )

    job.run(service_account=service_account)

    # Retrieve artifact paths
    pipeline_tasks = job.gca_resource.job_detail.task_details
    prepare_data_task = [pipeline_task for pipeline_task in pipeline_tasks if pipeline_task.task_name == 'prepare-data'][0]
    datasets_artifacts = {dataset: artifacts.artifacts[0].name for dataset, artifacts in prepare_data_task.outputs.items()}

    return datasets_artifacts


def get_worker_pool_specs(artifacts_resource_names: dict) -> list:
    args = []
    for artifact in ['X_train_output', 'y_train_output', 'X_val_output', 'y_val_output']:
        arg_name = artifact.replace('_output', '')
        arg_val = artifacts_resource_names[artifact].split('/')[-1]

        args.append(f'--{arg_name}={arg_val}')

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": None,
                "accelerator_count": None,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": 'europe-west3-docker.pkg.dev/ml-spec-demo2/blackfriday-pipeline/train_model_pipeline:latest',
                "args": args,
            }
        }
    ]

    return worker_pool_specs


def create_train_job(
    package_name: str,
    worker_pool_specs: list,
    staging_bucket: str
) -> aiplatform.CustomJob:
    custom_job = aiplatform.CustomJob(
        display_name=package_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_bucket,
        base_output_dir=f'{staging_bucket}/{package_name}'
    )

    return custom_job


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    init_aiplatform(args)

    artifacts_resource_names = run_prepare_data_pipeline(
        service_account=args.service_account,
        pipeline_root=args.pipeline_root,
        package_path=args.prepare_data_package_path,
    )

    worker_pool_specs = get_worker_pool_specs(artifacts_resource_names)
    train_model_package_name = args.train_model_package_path.split('.')[0]

    train_job = create_train_job(
        package_name=train_model_package_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=args.gcs_staging_bucket
    )

    parameters = {
        'n_estimators': hpt.IntegerParameterSpec(min=100, max=500, scale='linear'),
        'max_depth': hpt.IntegerParameterSpec(min=3, max=30, scale='linear'),
        'min_child_weight': hpt.IntegerParameterSpec(min=1, max=10, scale='linear'),
        'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='linear'),
    }

    metric = {'rmse': 'minimize'}


    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=f"hpo",
        custom_job=train_job,
        metric_spec=metric,
        parameter_spec=parameters,
        max_trial_count=2,
        parallel_trial_count=1,
    )

    hpt_job.run()

