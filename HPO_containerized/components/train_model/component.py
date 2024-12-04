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
    import os
    print(f'train model component environ: {os.environ}')

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