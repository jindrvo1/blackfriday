import argparse
import json

from google.cloud import aiplatform
from google.oauth2 import service_account
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(
    base_image='python:3.11',
    packages_to_install=[
        'pandas>=2.2.3',
        'git+https://github.com/jindrvo1/blackfriday',
        'google-cloud-storage>=2.18.2',
        'fsspec>=2024.10.0',
        'gcsfs>=2024.10.0',
    ]
)
def prepare_data(
    gcs_train_data_path: str,
    gcs_test_data_path: str,
    X_train_output: Output[Dataset],
    y_train_output: Output[Dataset],
    X_val_output: Output[Dataset],
    y_val_output: Output[Dataset],
    X_test_output: Output[Dataset],
):
    from tgmblackfriday import BlackFridayDataset

    dataset = BlackFridayDataset(gcs_train_data_path, gcs_test_data_path)
    dataset.preprocess_dfs(return_res=False)

    X_train, y_train, X_val, y_val, X_test = dataset.prepare_features_and_target(test_size=0.2, shuffle=True)

    X_train.to_csv(X_train_output.path, index=False)
    y_train.to_csv(y_train_output.path, index=False)
    X_val.to_csv(X_val_output.path, index=False)
    y_val.to_csv(y_val_output.path, index=False)
    X_test.to_csv(X_test_output.path, index=False)


@component(
    base_image='python:3.11',
    packages_to_install=[
        'pandas>=2.2.3',
        'xgboost>=2.1.2',
        'scikit-learn>=1.5.2',
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
    import pandas as pd
    from xgboost.sklearn import XGBRegressor

    X_train = pd.read_csv(X_train_input.path)
    y_train = pd.read_csv(y_train_input.path)

    X_val = pd.read_csv(X_val_input.path)
    y_val = pd.read_csv(y_val_input.path)

    model = XGBRegressor(
        n_estimators=n_estimators,
        objective=objective,
        eval_metric=eval_metric,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        early_stopping_rounds=10,
        seed=0
    )

    model = model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    joblib.dump(model, model_output.path)


@component(
    base_image='python:3.11',
    packages_to_install=[
        'pandas>=2.2.3',
        'xgboost>=2.1.2',
        'joblib>=1.4.2',
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
    base_image='python:3.11',
    packages_to_install=[
        'pandas>=2.2.3',
        'scikit-learn>=1.5.2',
    ]
)
def calc_metrics(
    y_true_input: Input[Dataset],
    y_pred_input: Input[Dataset],
    metrics_output: Output[Metrics],
):
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error
    )

    y_true = pd.read_csv(y_true_input.path)
    y_pred = pd.read_csv(y_pred_input.path)

    metrics_output.log_metric('rmse', root_mean_squared_error(y_true, y_pred))
    metrics_output.log_metric('mse', mean_squared_error(y_true, y_pred))
    metrics_output.log_metric('mae', mean_absolute_error(y_true, y_pred))


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
    data = prepare_data(
        gcs_train_data_path=gcs_train_data_path,
        gcs_test_data_path=gcs_test_data_path,
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
    )
    y_val_pred_job.set_display_name('predict-validation')

    y_train_pred_job = predict(
        model_input=model_job.outputs['model_output'],
        X_input=X_train,
    )
    y_train_pred_job.set_display_name('predict-train')

    y_test_pred_job = predict(
        model_input=model_job.outputs['model_output'],
        X_input=X_test,
    )
    y_test_pred_job.set_display_name('predict-test')

    val_metrics = calc_metrics(
        y_true_input=y_val,
        y_pred_input=y_val_pred_job.outputs['y_output'],
    )
    val_metrics.set_display_name('validation-metrics')

    train_metrics = calc_metrics(
        y_true_input=y_train,
        y_pred_input=y_train_pred_job.outputs['y_output'],
    )
    train_metrics.set_display_name('train-metrics')


def init_pipeline(
    package_path: str,
    gcs_train_data_path: str,
    gcs_test_data_path: str,
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


if __name__ == '__main__':
    with open('gcs_sa.json', 'r') as f:
        sa_key = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(sa_key)

    parser = argparse.ArgumentParser()
    parser.add_argument('--init_pipeline', action='store_true')
    parser.add_argument('--gcs_train_data_path', type=str, default='gs://blackfridaydataset/source_data/train.csv')
    parser.add_argument('--gcs_test_data_path', type=str, default='gs://blackfridaydataset/source_data/test.csv')
    parser.add_argument('--project_id', type=str, default='ml-spec-demo2')
    parser.add_argument('--region', type=str, default='europe-west3')
    parser.add_argument('--service_account', type=str, default='gcs-sa@ml-spec-demo2.iam.gserviceaccount.com')
    parser.add_argument('--pipeline_root', type=str, default='gs://blackfridaydataset/pipeline_root')
    parser.add_argument('--package_path', type=str, default='blackfriday_pipeline.yaml')
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)

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
            gcs_test_data_path=args.gcs_test_data_path
        )

        exit(0)

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