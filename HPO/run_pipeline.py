import argparse
from dotenv import dotenv_values
import json

from google.cloud import aiplatform, secretmanager
from google.oauth2 import service_account
from kfp import compiler, dsl, local
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

env_vars = dotenv_values('.env')

PROJECT_ID = env_vars['PROJECT_ID']
REGION = env_vars['REGION']
SERVICE_ACCOUNT = env_vars['SERVICE_ACCOUNT']
SA_SECRETS_NAME = env_vars['SA_SECRETS_NAME']
SECRETS_SERVICE_ACCOUNT = env_vars['SECRETS_SERVICE_ACCOUNT']
GCS_TRAIN_DATA_PATH = env_vars['GCS_TRAIN_DATA_PATH']
GCS_TEST_DATA_PATH = env_vars['GCS_TEST_DATA_PATH']
PIPELINE_ROOT = env_vars['PIPELINE_ROOT']
PACKAGE_PATH = env_vars['PACKAGE_PATH']
KFP_BASE_IMAGE = env_vars['KFP_BASE_IMAGE']


@component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=[
        'git+https://github.com/jindrvo1/blackfriday',
    ]
)
def load_and_validate_data(
    gcs_train_data_path: str,
    gcs_test_data_path: str,
    df_train_output: Output[Dataset],
    df_test_output: Output[Dataset],
):
    from tgmblackfriday import BlackFridayDataset

    dataset = BlackFridayDataset(gcs_train_data_path, gcs_test_data_path)
    dataset.validate_data()
    df_train, df_test = dataset.get_dfs()

    df_train.to_csv(df_train_output.path, index=False)
    df_test.to_csv(df_test_output.path, index=False)


@component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=[
        'git+https://github.com/jindrvo1/blackfriday',
    ]
)
def prepare_data(
    df_train_input: Input[Dataset],
    df_test_input: Input[Dataset],
    X_train_output: Output[Dataset],
    y_train_output: Output[Dataset],
    X_val_output: Output[Dataset],
    y_val_output: Output[Dataset],
    X_test_output: Output[Dataset],
):
    from tgmblackfriday import BlackFridayDataset

    dataset = BlackFridayDataset(df_train_input.path, df_test_input.path)
    dataset.preprocess_dfs(return_res=False)

    X_train, y_train, X_val, y_val, X_test = dataset.prepare_features_and_target(test_size=0.2, shuffle=True)

    X_train.to_csv(X_train_output.path, index=False)
    y_train.to_csv(y_train_output.path, index=False)
    X_val.to_csv(X_val_output.path, index=False)
    y_val.to_csv(y_val_output.path, index=False)
    X_test.to_csv(X_test_output.path, index=False)


@component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=[
        'pandas>=2.2.3',
        'xgboost>=2.1.2',
        'scikit-learn>=1.5.2',
        'git+https://github.com/jindrvo1/blackfriday',
        'cloudml-hypertune==0.1.0.dev6',
    ]
)
def train_model(
    X_train_input: Input[Dataset],
    y_train_input: Input[Dataset],
    X_val_input: Input[Dataset],
    y_val_input: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 300,
    max_depth: int = 6,
    min_child_weight: int = 1,
    learning_rate: float = 0.1,
    objective: str = 'reg:squarederror',
    eval_metric: str = 'rmse',
):
    import joblib
    import hypertune
    import pandas as pd
    from xgboost.sklearn import XGBRegressor
    from tgmblackfriday import ReportValRmseCallback


    X_train = pd.read_csv(X_train_input.path)
    y_train = pd.read_csv(y_train_input.path)

    X_val = pd.read_csv(X_val_input.path)
    y_val = pd.read_csv(y_val_input.path)

    hpt = hypertune.HyperTune()
    report_val_rmse_callback = ReportValRmseCallback(hpt=hpt)

    model = XGBRegressor(
        n_estimators=n_estimators,
        objective=objective,
        eval_metric=eval_metric,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        early_stopping_rounds=10,
        callbacks=[report_val_rmse_callback],
        seed=0
    )

    model = model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    joblib.dump(model, model_output.path)


@component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=[
        'pandas>=2.2.3',
        'xgboost>=2.1.2',
        'joblib>=1.4.2',
        'git+https://github.com/jindrvo1/blackfriday',
        'cloudml-hypertune==0.1.0.dev6',
    ]
)
def predict(
    model_input: Input[Model],
    X_input: Input[Dataset],
    y_output: Output[Dataset],
):
    import joblib
    import pandas as pd

    model = joblib.load(model_input.path)

    X = pd.read_csv(X_input.path)
    y_pred = model.predict(X)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv(y_output.path, index=False)


@component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=[
        'pandas>=2.2.3',
        'scikit-learn>=1.5.2',
    ]
)
def calc_metrics(
    y_true_input: Input[Dataset],
    y_pred_input: Input[Dataset],
    tag_prefix: str,
    metrics_output: Output[Metrics],
):
    import json
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error
    )

    y_true = pd.read_csv(y_true_input.path)
    y_pred = pd.read_csv(y_pred_input.path)

    metrics_dict = {
        f'{tag_prefix}_rmse': root_mean_squared_error(y_true, y_pred),
        f'{tag_prefix}_mse': mean_squared_error(y_true, y_pred),
        f'{tag_prefix}_mae': mean_absolute_error(y_true, y_pred)
    }

    with open(metrics_output.path, 'w') as f:
        json.dump(metrics_dict, f)

    metrics_output.log_metric(f'{tag_prefix}_rmse', metrics_dict[f'{tag_prefix}_rmse'])
    metrics_output.log_metric(f'{tag_prefix}_mse', metrics_dict[f'{tag_prefix}_mse'])
    metrics_output.log_metric(f'{tag_prefix}_mae', metrics_dict[f'{tag_prefix}_mae'])


# @component(
#     base_image=KFP_BASE_IMAGE,
#     packages_to_install=[
#         'cloudml-hypertune==0.1.0.dev6'
#     ]
# )
# def log_metrics(metrics_input: Input[Metrics], tag_prefix: str):
#     import hypertune
#     import json

#     with open(metrics_input.path) as f:
#         metrics = json.load(f)

#     rmse = metrics.get(f'{tag_prefix}_rmse')
#     mse = metrics.get(f'{tag_prefix}_mse')
#     mae = metrics.get(f'{tag_prefix}_mae')

#     hpt = hypertune.HyperTune()
#     hpt.report_hyperparameter_tuning_metric(
#         hyperparameter_metric_tag=f"{tag_prefix}_rmse",
#         metric_value=rmse
#     )
#     hpt.report_hyperparameter_tuning_metric(
#         hyperparameter_metric_tag=f"{tag_prefix}_mse",
#         metric_value=mse
#     )
#     hpt.report_hyperparameter_tuning_metric(
#         hyperparameter_metric_tag=f"{tag_prefix}_mae",
#         metric_value=mae
#     )


@dsl.pipeline()
def blackfriday_pipeline(
    gcs_train_data_path: str,
    gcs_test_data_path: str,
    n_estimators: int = 300,
    max_depth: int = 6,
    min_child_weight: int = 1,
    learning_rate: float = 0.1,
    objective: str = 'reg:squarederror',
    eval_metric: str = 'rmse',
):
    load_data_job = load_and_validate_data(
        gcs_train_data_path=gcs_train_data_path,
        gcs_test_data_path=gcs_test_data_path
    )

    data = prepare_data(
        df_train_input=load_data_job.outputs['df_train_output'],
        df_test_input=load_data_job.outputs['df_test_output'],
    )

    X_train = data.outputs['X_train_output']
    y_train = data.outputs['y_train_output']
    X_val = data.outputs['X_val_output']
    y_val = data.outputs['y_val_output']
    X_test = data.outputs['X_test_output']

    model_job = train_model(
        X_train_input=X_train,
        y_train_input=y_train,
        X_val_input=X_val,
        y_val_input=y_val,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric=eval_metric
    )

    y_val_pred_job = predict(
        model_input=model_job.outputs['model_output'],
        X_input=X_val,
    ).set_display_name('predict-validation')

    y_train_pred_job = predict(
        model_input=model_job.outputs['model_output'],
        X_input=X_train,
    ).set_display_name('predict-train')

    y_test_pred_job = predict(
        model_input=model_job.outputs['model_output'],
        X_input=X_test,
    ).set_display_name('predict-test')

    val_metrics = calc_metrics(
        y_true_input=y_val,
        y_pred_input=y_val_pred_job.outputs['y_output'],
        tag_prefix='val',
    ).set_display_name('validation-metrics')

    train_metrics = calc_metrics(
        y_true_input=y_train,
        y_pred_input=y_train_pred_job.outputs['y_output'],
        tag_prefix='train',
    ).set_display_name('train-metrics')

    # log_val_metrics_to_hypertune = log_metrics(
    #     metrics_input=val_metrics.outputs['metrics_output'],
    #     tag_prefix='val',
    # ).set_display_name('log-val-metrics-to-hypertune')

    # log_train_metrics_to_hypertune = log_metrics(
    #     metrics_input=train_metrics.outputs['metrics_output'],
    #     tag_prefix='train',
    # ).set_display_name('log-train-metrics-to-hypertune')


def init_pipeline(
    package_path: str,
    gcs_train_data_path: str,
    gcs_test_data_path: str
):
    compiler.Compiler().compile(
        pipeline_func=blackfriday_pipeline,
        package_path=package_path,
        pipeline_parameters={
            'gcs_train_data_path': gcs_train_data_path,
            'gcs_test_data_path': gcs_test_data_path
        },
    )


def run_pipeline(
    service_account: str,
    pipeline_root: str,
    package_path: str,
    hyperparameters: dict,
):
    job = aiplatform.PipelineJob(
        display_name=package_path.split('.')[0],
        template_path=package_path,
        pipeline_root=pipeline_root,
        parameter_values=hyperparameters,
        enable_caching=True,
    )

    job.run(service_account=service_account)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--init_pipeline', action='store_true')
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--sa_secrets_name', type=str, default=SA_SECRETS_NAME)
    parser.add_argument('--gcs_train_data_path', type=str, default=GCS_TRAIN_DATA_PATH)
    parser.add_argument('--gcs_test_data_path', type=str, default=GCS_TEST_DATA_PATH)
    parser.add_argument('--project_id', type=str, default=PROJECT_ID)
    parser.add_argument('--region', type=str, default=REGION)
    parser.add_argument('--service_account', type=str, default=SERVICE_ACCOUNT)
    parser.add_argument('--pipeline_root', type=str, default=PIPELINE_ROOT)
    parser.add_argument('--package_path', type=str, default=PACKAGE_PATH)
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    return parser


def get_credentials(secret_name: str) -> service_account.Credentials:
    with open(SECRETS_SERVICE_ACCOUNT, 'r') as f:
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

    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'learning_rate': args.learning_rate
    }

    if args.init_pipeline:
        init_pipeline(
            package_path=args.package_path,
            gcs_train_data_path=args.gcs_train_data_path,
            gcs_test_data_path=args.gcs_test_data_path,
        )
        print('Pipeline initialized')
        exit(0)

    if args.run_locally:
        local.init(runner=local.DockerRunner())

        blackfriday_pipeline(
            gcs_train_data_path=args.gcs_train_data_path,
            gcs_test_data_path=args.gcs_test_data_path,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            learning_rate=args.learning_rate,
            objective='reg:squarederror',
            eval_metric='rmse',
        )
        exit(0)

    sa_secrets_name = f'projects/{args.project_id}/secrets/{args.sa_secrets_name}/versions/latest'
    credentials = get_credentials(sa_secrets_name)

    aiplatform.init(
        project=args.project_id,
        location=args.region,
        credentials=credentials
    )

    run_pipeline(
        service_account=args.service_account,
        pipeline_root=args.pipeline_root,
        package_path=args.package_path,
        hyperparameters=hyperparameters
    )
