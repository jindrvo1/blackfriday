from dotenv import dotenv_values
import os

from kfp.dsl import component, Output, Dataset


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