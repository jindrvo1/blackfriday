from dotenv import dotenv_values
import os

from kfp.dsl import component, Input, Output, Dataset, Metrics


component_env_path = os.path.join(os.path.dirname(__file__), '.env')
component_env = dotenv_values(component_env_path)

KFP_BASE_IMAGE = component_env['KFP_BASE_IMAGE']
TARGET_IMAGE = component_env['TARGET_IMAGE']


@component(
    base_image=KFP_BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[
        'pandas>=2.2.3',
        'scikit-learn>=1.5.2',
        'cloudml-hypertune==0.1.0.dev6',
        'google-cloud-logging>=3.11.3'
    ]
)
def evaluate(
    y_true_input: Input[Dataset],
    y_pred_input: Input[Dataset],
    tag_prefix: str,
    metrics_output: Output[Metrics],
) -> float:
    import json
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error
    )
    import hypertune
    import google.cloud.logging
    import logging

    import os
    print(f'run_hpo.py environ: {os.environ}')
    logging_client = google.cloud.logging.Client()
    logging_client.setup_logging()
    logging.warning(f'run_hpo.py environ: {os.environ}')

    y_true = pd.read_csv(y_true_input.path)
    y_pred = pd.read_csv(y_pred_input.path)

    metrics_dict = {
        f'{tag_prefix}rmse': root_mean_squared_error(y_true, y_pred),
        f'{tag_prefix}mse': mean_squared_error(y_true, y_pred),
        f'{tag_prefix}mae': mean_absolute_error(y_true, y_pred)
    }

    hpt = hypertune.HyperTune()

    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='rmse',
        metric_value=metrics_dict['rmse'],
        global_step=100
    )

    metrics_output.log_metric(f'{tag_prefix}rmse', metrics_dict[f'{tag_prefix}rmse'])
    metrics_output.log_metric(f'{tag_prefix}mse', metrics_dict[f'{tag_prefix}mse'])
    metrics_output.log_metric(f'{tag_prefix}mae', metrics_dict[f'{tag_prefix}mae'])

    with open(metrics_output.path, 'w') as f:
        json.dump(metrics_dict, f)



