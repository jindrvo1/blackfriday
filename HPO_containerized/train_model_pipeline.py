import argparse
import re
from dotenv import dotenv_values
import json

from kfp import compiler, dsl, local
from kfp.dsl import Dataset
from google.cloud import aiplatform, secretmanager
from google.oauth2 import service_account

from components.train_model.component import train_model
from components.predict.component import predict
from components.evaluate.component import evaluate


env_vars = dotenv_values('.env')

TRAIN_MODEL_PACKAGE_PATH = env_vars['TRAIN_MODEL_PACKAGE_PATH']
PROJECT_ID = env_vars['PROJECT_ID']
REGION = env_vars['REGION']
PIPELINE_ROOT = env_vars['PIPELINE_ROOT']
SERVICE_ACCOUNT = env_vars['SERVICE_ACCOUNT']
SA_SECRETS_NAME = env_vars['SA_SECRETS_NAME']
SECRETS_SERVICE_ACCOUNT = env_vars['SECRETS_SERVICE_ACCOUNT']


@dsl.pipeline
def train_model_pipeline(
    X_train_input: Dataset,
    y_train_input: Dataset,
    X_val_input: Dataset,
    y_val_input: Dataset,
    n_estimators: int = 300,
    max_depth: int = 6,
    min_child_weight: int = 1,
    learning_rate: float = 0.1,
    objective: str = 'reg:squarederror',
    eval_metric: str = 'rmse',
):
    train_model_job = train_model(
        X_train_input=X_train_input,
        y_train_input=y_train_input,
        X_val_input=X_val_input,
        y_val_input=y_val_input,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric=eval_metric,
    )

    predict_job = predict(
        model_input=train_model_job.outputs['model_output'],
        X_input=X_val_input,
    )

    evaluate_job = evaluate(
        y_true_input=y_val_input,
        y_pred_input=predict_job.outputs['y_output'],
        tag_prefix='',
    )


def run_pipeline(
    service_account: str,
    pipeline_root: str,
    package_path: str,
    data: dict,
    hyperparameters: dict,
):
    job = aiplatform.PipelineJob(
        display_name=package_path.split('.')[0],
        template_path=package_path,
        pipeline_root=pipeline_root,
        input_artifacts=data,
        parameter_values=hyperparameters,
        enable_caching=True,
    )

    job.run(service_account=service_account)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--sa_secrets_name', type=str, default=SA_SECRETS_NAME)
    parser.add_argument('--secrets_service_account', type=str, default=SECRETS_SERVICE_ACCOUNT)
    parser.add_argument('--project_id', type=str, default=PROJECT_ID)
    parser.add_argument('--region', type=str, default=REGION)
    parser.add_argument('--service_account', type=str, default=SERVICE_ACCOUNT)
    parser.add_argument('--pipeline_root', type=str, default=PIPELINE_ROOT)
    parser.add_argument('--train_model_package_path', type=str, default=TRAIN_MODEL_PACKAGE_PATH)

    parser.add_argument('--X_train', type=str, required=True)
    parser.add_argument('--y_train', type=str, required=True)
    parser.add_argument('--X_val', type=str, required=True)
    parser.add_argument('--y_val', type=str, required=True)

    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)

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


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=train_model_pipeline,
        package_path=args.train_model_package_path
    )

    # sa_secrets_name = f'projects/{args.project_id}/secrets/{args.sa_secrets_name}/versions/latest'
    # credentials = get_credentials(args.secrets_service_account, sa_secrets_name)

    aiplatform.init(
        location=args.region
    )

    data = {
        'X_train_input': args.X_train,
        'y_train_input': args.y_train,
        'X_val_input': args.X_val,
        'y_val_input': args.y_val,
    }

    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'learning_rate': args.learning_rate
    }

    run_pipeline(
        service_account=args.service_account,
        pipeline_root=args.pipeline_root,
        package_path=args.train_model_package_path,
        data=data,
        hyperparameters=hyperparameters,
    )