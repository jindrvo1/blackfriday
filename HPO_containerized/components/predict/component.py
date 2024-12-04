from dotenv import dotenv_values
import os

from kfp.dsl import component, Input, Output, Dataset, Model


component_env_path = os.path.join(os.path.dirname(__file__), '.env')
component_env = dotenv_values(component_env_path)

KFP_BASE_IMAGE = component_env['KFP_BASE_IMAGE']
TARGET_IMAGE = component_env['TARGET_IMAGE']


@component(
    base_image=KFP_BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[
        'pandas>=2.2.3',
        'xgboost>=2.1.2',
        'joblib>=1.4.2',
        'git+https://github.com/jindrvo1/blackfriday',
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
