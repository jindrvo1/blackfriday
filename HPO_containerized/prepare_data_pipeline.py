import argparse
from dotenv import dotenv_values
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import Dataset

from components.load_and_validate_data.component import load_and_validate_data
from components.prepare_data.component import prepare_data


env_vars = dotenv_values('.env')

GCS_TRAIN_DATA_PATH = env_vars['GCS_TRAIN_DATA_PATH']
GCS_TEST_DATA_PATH = env_vars['GCS_TEST_DATA_PATH']
PREPARE_DATA_PACKAGE_PATH = env_vars['PREPARE_DATA_PACKAGE_PATH']


@dsl.pipeline()
def prepare_data_pipeline(
    gcs_train_data_path: str,
    gcs_test_data_path: str,
) -> NamedTuple(
        'pipeline_outputs',
        X_train=Dataset,
        y_train=Dataset,
        X_val=Dataset,
        y_val=Dataset,
        X_test=Dataset
    ):
    load_data_job = load_and_validate_data(
        gcs_train_data_path=gcs_train_data_path,
        gcs_test_data_path=gcs_test_data_path,
    )

    prepare_data_job = prepare_data(
        df_train_input=load_data_job.outputs['df_train_output'],
        df_test_input=load_data_job.outputs['df_test_output'],
    )

    X_train = prepare_data_job.outputs['X_train_output']
    y_train = prepare_data_job.outputs['y_train_output']
    X_val = prepare_data_job.outputs['X_val_output']
    y_val = prepare_data_job.outputs['y_val_output']
    X_test = prepare_data_job.outputs['X_test_output']

    pipeline_outputs = NamedTuple('pipeline_outputs', X_train=Dataset, y_train=Dataset, X_val=Dataset, y_val=Dataset, X_test=Dataset)

    return pipeline_outputs(X_train, y_train, X_val, y_val, X_test)


def compile_prepare_data_pipeline(
    package_path: str,
    gcs_train_data_path: str,
    gcs_test_data_path: str
):
    print('Compiling pipeline...')
    compiler.Compiler().compile(
        pipeline_func=prepare_data_pipeline,
        package_path=package_path,
        pipeline_parameters={
            'gcs_train_data_path': gcs_train_data_path,
            'gcs_test_data_path': gcs_test_data_path
        },
    )

    print('Prepare data pipeline compiled')


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcs_train_data_path', type=str, default=GCS_TRAIN_DATA_PATH)
    parser.add_argument('--gcs_test_data_path', type=str, default=GCS_TEST_DATA_PATH)
    parser.add_argument('--prepare_data_package_path', type=str, default=PREPARE_DATA_PACKAGE_PATH)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    compile_prepare_data_pipeline(
        package_path=args.prepare_data_package_path,
        gcs_train_data_path=args.gcs_train_data_path,
        gcs_test_data_path=args.gcs_test_data_path
    )