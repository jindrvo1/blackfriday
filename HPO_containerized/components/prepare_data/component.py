from dotenv import dotenv_values
import os

from kfp.dsl import component, Input, Output, Dataset


component_env_path = os.path.join(os.path.dirname(__file__), '.env')
component_env = dotenv_values(component_env_path)

KFP_BASE_IMAGE = component_env['KFP_BASE_IMAGE']
TARGET_IMAGE = component_env['TARGET_IMAGE']


@component(
    base_image=KFP_BASE_IMAGE,
    target_image=TARGET_IMAGE,
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
