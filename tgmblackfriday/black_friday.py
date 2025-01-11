from enum import Enum
from pathlib import Path

import xgboost
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from google.cloud import storage


class EncodingType(Enum):
    BINARY = 0
    ORDINAL = 1
    ONE_HOT = 2


class ReportValRmseCallback(xgboost.callback.TrainingCallback):
    def __init__(self, hpt, metric):
        self.hpt = hpt
        self.metric = metric
        self.iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        self.iteration = epoch

        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric,
            metric_value=evals_log['validation_0'][self.metric][-1],
            global_step=self.iteration
        )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    min_child_weight: int,
    learning_rate: float,
    objective: str,
    eval_metric: str,
    hypertune_instance: object = None,
) -> XGBRegressor:
    if hypertune_instance:
        report_val_rmse_callback = ReportValRmseCallback(
            hpt=hypertune_instance,
            metric=eval_metric
        )

        model = XGBRegressor(
            n_estimators=n_estimators,
            objective=objective,
            eval_metric=eval_metric,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            early_stopping_rounds=10,
            seed=0,
            callbacks=[report_val_rmse_callback]
        )

        model = model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        hypertune_instance.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=eval_metric,
            metric_value=model.evals_result_['validation_0'][eval_metric][-1],
            global_step=len(model.evals_result_['validation_0'][eval_metric])
        )
    else:
        model = XGBRegressor(
            n_estimators=n_estimators,
            objective=objective,
            eval_metric=eval_metric,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            early_stopping_rounds=10,
            seed=0,
        )

        model = model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

    return model


class ProductCategoriesEncoder:
    product_category_cols: list[str] = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    def __init__(self, encoding: EncodingType, product_category_cols: list[str] = product_category_cols):
        self.encoding = encoding
        self.product_category_cols = product_category_cols


    def fit(self, df: pd.DataFrame) -> None:
        categories = set()
        for product_category in self.product_category_cols:
            categories |= set(df[product_category].dropna().unique())

        categories = sorted(list(categories))

        self._product_categories = categories


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encoding.value == EncodingType.ONE_HOT.value:
            df_products = df.drop_duplicates(subset=['Product_ID'])
            df_products = df_products.set_index('Product_ID')
            df_products = df_products[self.product_category_cols]

            df_products_encoded = df_products[self.product_category_cols].apply(self._encode_product_row_one_hot, axis=1)
            df_products_encoded = df_products_encoded.loc[:, df_products_encoded.columns[1:]]

            df = df.drop(columns=self.product_category_cols).merge(df_products_encoded, left_on='Product_ID', right_index=True)
        elif self.encoding.value == EncodingType.ORDINAL.value:
            # Product categories already preserve ordinal relationship.
            # Keeping this here for clarity.
            ...

        return df


    def _encode_product_row_one_hot(self, row: pd.Series) -> pd.Series:
        product_categories = row.dropna().values
        encoded_row = pd.Series({
            f'Product_Category_{category}': 1 if category in product_categories else 0 for category in self._product_categories
        })

        return encoded_row


class BlackFridayDataset:
    _age_encoder: OrdinalEncoder | OneHotEncoder | None = None
    _city_category_encoder: OrdinalEncoder | OneHotEncoder | None = None
    _occupation_encoder: OrdinalEncoder | OneHotEncoder | None = None
    _stay_in_current_city_years_encoder: OrdinalEncoder | OneHotEncoder | None = None
    _product_category_encoder: ProductCategoriesEncoder | None = None

    demographic_cols: list[str] = [
        'Gender', 'Age', 'City_Category', 'Occupation',
        'Marital_Status', 'Stay_In_Current_City_Years'
    ]

    product_category_cols: list[str] = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    feature_encodings: dict[str, EncodingType] = {
        'Age': EncodingType.ONE_HOT,
        'Occupation': EncodingType.ORDINAL,
        'Gender': EncodingType.BINARY,
        'City_Category': EncodingType.ONE_HOT,
        'Product_Category': EncodingType.ONE_HOT
    }


    def __init__(self, path: str, test_path: str = None):
        if path.startswith('gs://'):
            self.gcs_client = storage.Client.create_anonymous_client()

            self.df = pd.read_csv(path)
            self.df_test = pd.read_csv(test_path) if test_path else None
        else:
            self.df = self._load_data_file(path)
            self.df_test = self._load_data_file(test_path) if test_path else None


    def validate_data(self):
        target_col = 'Purchase'

        # Check for presence of all expected columns
        cols_to_check = ['User_ID', 'Product_ID'] + self.demographic_cols + self.product_category_cols
        for col in cols_to_check:
            assert col in self.df.columns, f"Column `{col}` is missing from the train dataset"
            assert col in self.df_test.columns, f"Column `{col}` is missing from the test dataset"

        assert target_col in self.df.columns, f"Column `{target_col}` is missing from the train dataset"

        # Check for unexpected missing values
        cols_to_check = self.demographic_cols + [self.product_category_cols[0]]
        for col in cols_to_check:
            assert self.df[col].isna().sum() == 0, f"Missing values in column `{col}` of the train dataset"
            assert self.df_test[col].isna().sum() == 0, f"Missing values in column `{col}` of the test dataset"

        assert self.df[target_col].isna().sum() == 0, f"Missing values in column `{target_col}` of the test dataset"

        # Check that the test dataset does not have values not present in the train dataset
        # Product categories are handled separately
        cols_to_check = self.demographic_cols
        for col in cols_to_check:
            df_col_unique = set(self.df[col].unique())
            df_test_col_unique = set(self.df_test[col].unique())
            diff = df_test_col_unique - df_col_unique
            assert len(diff) == 0, f"Values `{', '.join(diff)}` in column `{col}` " + \
                                        f"of the test dataset are not present in the train dataset"

        # Check that the test dataset does not have values not present in the train dataset for product categories
        df_categories = np.unique(self.df[self.product_category_cols].values.flatten())
        df_categories = set(df_categories[~np.isnan(df_categories)])

        df_test_categories = np.unique(self.df_test[self.product_category_cols].values.flatten())
        df_test_categories = set(df_test_categories[~np.isnan(df_test_categories)])

        diff = df_test_categories - df_categories
        assert len(diff) == 0, f"Values `{', '.join(diff)}` in `Product_Category_*` columns " + \
                                    f"of the test dataset are not present in the train dataset"


    def preprocess_dfs(self, encodings: dict[str, EncodingType] = None, return_res: bool = True) -> None | tuple[pd.DataFrame, pd.DataFrame | None]:
        """Processes the features in the loaded dataframes to be ready to be used
        in a model. Specific processing steps for each feature are further
        explained in the relevant functions

        Args:
            return_res (bool): Whether to return the processed dataframe(s)

        Returns:
            None | tuple[pd.DataFrame, pd.DataFrame | None]: If `return_res`
                is True, returns the processed dataframe(s), depending on whether
                the test dataset was loaded
        """

        encodings = self.feature_encodings if encodings is None else encodings

        cols_to_drop = set(self.demographic_cols + ['Product_Category']) - set(encodings.keys())
        self.df = self.df.drop(columns=cols_to_drop)
        self.df_test = self.df_test.drop(columns=cols_to_drop) if self.df_test is not None else None

        self.df_encoded = self.df.copy()
        self.df_test_encoded = self.df_test.copy() if self.df_test is not None else None
        self.df_encoded = self._preprocess_df(
            self.df_encoded,
            encodings
        )

        self.df_test_encoded = self._preprocess_df(
            self.df_test_encoded,
            encodings
        ) if self.df_test_encoded is not None else None

        return (self.df_encoded, self.df_test_encoded) if return_res else None


    def get_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Returns the raw (unprocessed) dataframe(s)

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: If test dataset was loaded,
                returns both dataframes in their raw form, otherwise returns only the
                training dataframe
        """
        return (self.df, self.df_test)


    def get_dfs_encoded(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Returns the processed dataframe(s)

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: If test dataset was loaded,
                returns both dataframes in their processed form, otherwise returns only
                the training dataframe
        """
        return (self.df_encoded, self.df_test_encoded)


    def get_features_and_target(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Returns the features and target from the loaded dataframe. If train-validation
        split was performed, the validation set is also returned

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
                A tuple of the following dataframes in this order:
                    - Training features
                    - Training target
                    - Validation features if `test_size` is not None, otherwise None
                    - Validation target if `test_size` is not None, otherwise None
        """
        return (self.X_train, self.y_train, self.X_val, self.y_val)


    def prepare_features_and_target(
        self,
        test_size: float = 0.3,
        shuffle: bool = True,
        target_col: str = 'Purchase',
        cols_to_drop: list[str] = ['User_ID', 'Product_ID'],
        return_res: bool = True,
    ) -> None | tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | None,
            pd.DataFrame | None,
            pd.DataFrame | None,
            pd.DataFrame | None,
            pd.DataFrame | None,
        ]:
        """Splits the loaded dataframe into train and validation sets and
        extracts the features, except for the `cols_to_drop` columns, and
        the target column from the dataframe, returned as a tuple

        Args:
            test_size (float, optional): The proportion of the dataset to be used
                for validation. Defaults to 0.2. If None is passed, no splitting
                will be performed
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True
            target_col (str, optional): The name of the target column.
                Defaults to 'Purchase'
            cols_to_drop (list[str], optional): The names of the columns to drop.
                Defaults to `['User_ID', 'Product_ID', 'Purchase']`
            return_res (bool): Whether to return the resulting dataframes.
                Defaults to True

        Returns:
            None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
                If `return_res` is True, returns a tuple of the following dataframes in this order:
                    - Training features
                    - Training target
                    - Validation features if `test_size` is not None, otherwise None
                    - Validation target if `test_size` is not None, otherwise None
                    - Test features if test dataset was loaded
                Otherwise, returns None

        """
        X = self.df_encoded.drop(columns=cols_to_drop + [target_col])
        y = self.df_encoded[target_col]

        if test_size:
            self.X_train, X_val_test, self.y_train, y_val_test = train_test_split(
                X, y,
                test_size=test_size,
                shuffle=shuffle
            )
            self.X_val, self.X_test_ind, self.y_val, self.y_test_ind = train_test_split(
                X_val_test, y_val_test,
                test_size=0.5,
                shuffle=shuffle
            )
        else:
            self.X_train = X.sample(frac=1) if shuffle else X
            self.y_train = y.sample(frac=1) if shuffle else y
            self.X_val, self.y_val = None, None

        self.X_test = self.df_test_encoded.drop(columns=cols_to_drop) if self.df_test_encoded is not None else None

        return (self.X_train, self.y_train, self.X_val, self.y_val, self.X_test_ind, self.y_test_ind, self.X_test) if return_res else None


    def _preprocess_df(self, df: pd.DataFrame, encodings: dict[str, EncodingType]) -> pd.DataFrame:
        """Processes the features in the loaded dataframe to be ready to be used
        in a model. Specific processing steps for each feature are further
        explained in the relevant functions

        Args:
            df (pd.DataFrame): The dataframe to process
            process_feature_funcs (list[callable]): A list of functions that
                process the features in the dataframe

        Returns:
            pd.DataFrame: The processed dataframe
        """
        col_func_mapping = {
            'Gender': self._process_gender_col,
            'Age': self._process_age_col,
            'City_Category': self._process_city_category_col,
            'Stay_In_Current_City_Years': self._process_stay_in_current_city_years_col,
            'Occupation': self._process_occupation_col,
            'Product_Category': self._process_product_category_cols
        }

        for col, encoding in encodings.items():
            if col in col_func_mapping.keys():
                col_func = col_func_mapping[col]
                df = col_func(df, encoding)

        return df


    def _load_data_file(self, path: str) -> pd.DataFrame:
        """Loads the data file at the specified path into a pandas dataframe

        Args:
            path (str): The path to the data file as a string

        Raises:
            ValueError: If the path is incorrect or the file does not exist

        Returns:
            pd.DataFrame: The file loaded into a pandas dataframe
        """
        path = Path(path)

        if path.is_file():
            return pd.read_csv(path,)
        else:
            raise ValueError(f"Incorrect path to the dataset: '{path}'")


    def _process_gender_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Maps the 'F' and 'M' strings in the 'Gender' column of the
        `df` dataframe to 0 and 1 respectively

        Args:
            df (pd.DataFrame): The dataframe with the 'Gender' column
                populated by the 'M' and 'F' strings
            encoding (EncodingType): The encoding type to use. Can be either
                `EncodingType.ORDINAL` or `EncodingType.ONE_HOT`. Not used
                in this function, kept for consistency with other functions.

        Returns:
            pd.DataFrame: The dataframe with the 'Gender' column's
                values mapped to 0 and 1
        """
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

        return df

    def _prepare_col_encoder(
        self,
        df: pd.DataFrame,
        encoder_type: EncodingType,
        col: str
    ) -> OneHotEncoder | OrdinalEncoder:
        categories = sorted(df[col].unique())

        if encoder_type.value == EncodingType.ORDINAL.value:
            encoder = OrdinalEncoder(categories=[categories], dtype=int)
            encoder.fit(df[[col]])

        elif encoder_type.value == EncodingType.ONE_HOT.value:
            encoder = OneHotEncoder(drop='first', categories=[categories], dtype=int)
            encoder.fit(df[[col]])

        else:
            raise ValueError(f"Invalid encoding type: {encoder_type}")

        return encoder


    def _encode_col(self, df: pd.DataFrame, encoder: OneHotEncoder | OrdinalEncoder, col: str) -> pd.DataFrame:
        if isinstance(encoder, OrdinalEncoder):
            df[col] = encoder.transform(df[[col]])

        elif isinstance(encoder, OneHotEncoder):
            col_names = [f'{col}_{category}' for category in encoder.categories[0][1:]]
            df[col_names] = encoder.transform(df[[col]]).toarray()
            df = df.drop(columns=[col])

        else:
            raise ValueError(f"Invalid encoder type: {type(encoder)}")

        return df


    def _process_age_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Encodes the `Age` column of the `df` dataframe to either ordinal
        integer values or one-hot encoded representation depending on the `encoding`
        parameter.

        Args:
            df (pd.DataFrame): The dataframe with the 'Age' column populated
                with age categories represented by integer values
            encoding (EncodingType): The encoding type to use. Can be either
                `EncodingType.ORDINAL` or `EncodingType.ONE_HOT`

        Returns:
            pd.DataFrame: The dataframe with the 'Age' column replaced
                by ordinal integer values or one-hot encoded representation
        """
        if not self._age_encoder:
            self._age_encoder = self._prepare_col_encoder(df, encoding, 'Age')

        return self._encode_col(df, self._age_encoder, 'Age')


    def _process_city_category_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Encodes the `City_Category` column of the `df` dataframe to either ordinal
        integer values or one-hot encoded representation depending on the `encoding`
        parameter.

        Args:
            df (pd.DataFrame): The dataframe with the 'City_Category' column populated
                with city categories represented by integer values
            encoding (EncodingType): The encoding type to use. Can be either
                `EncodingType.ORDINAL` or `EncodingType.ONE_HOT`

        Returns:
            pd.DataFrame: The dataframe with the 'City_Category' column replaced
                by ordinal integer values or one-hot encoded representation
        """
        if not self._city_category_encoder:
            self._city_category_encoder = self._prepare_col_encoder(df, encoding, 'City_Category')

        return self._encode_col(df, self._city_category_encoder, 'City_Category')


    def _process_stay_in_current_city_years_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Encodes the `Stay_In_Current_City_Years` column of the `df` dataframe to either ordinal
        integer values or one-hot encoded representation depending on the `encoding`
        parameter.

        Args:
            df (pd.DataFrame): The dataframe with the 'Stay_In_Current_City_Years' column
                populated with the column's categories represented by integer values
            encoding (EncodingType): The encoding type to use. Can be either
                `EncodingType.ORDINAL` or `EncodingType.ONE_HOT`

        Returns:
            pd.DataFrame: The dataframe with the 'Stay_In_Current_City_Years' column
                replaced by ordinal integer values or one-hot encoded representation
        """
        if not self._stay_in_current_city_years_encoder:
            self._stay_in_current_city_years_encoder = self._prepare_col_encoder(df, encoding, 'Stay_In_Current_City_Years')

        return self._encode_col(df, self._stay_in_current_city_years_encoder, 'Stay_In_Current_City_Years')

    def _process_occupation_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Encodes the `Occupation` column of the `df` dataframe to either ordinal
        integer values or one-hot encoded representation depending on the `encoding`
        parameter.

        Args:
            df (pd.DataFrame): The dataframe with the 'Occupation' column populated
                with occupation categories represented by integer values
            encoding (EncodingType): The encoding type to use. Can be either
                `EncodingType.ORDINAL` or `EncodingType.ONE_HOT`

        Returns:
            pd.DataFrame: The dataframe with the 'Occupation' column replaced
                by ordinal integer values or one-hot encoded representation
        """
        if not self._occupation_encoder:
            self._occupation_encoder = self._prepare_col_encoder(df, encoding, 'Occupation')

        return self._encode_col(df, self._occupation_encoder, 'Occupation')


    def _process_product_category_cols(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """One-hot encodes the `Product_Category_*` columns of the `df` dataframe.
        The original `Product_Category_*` columns are dropped in favour of its encoded
        representation in N new columns where N is the number of different product
        categories

        Args:
            df (pd.DataFrame): The dataframe with the 'Product_Category_*' columns populated
                with product categories represented by integer values

        Returns:
            pd.DataFrame: The dataframe with the 'Product_Category_*' columns replaced
                by N columns representing its one-hot encoded representation.
                The output column names are `Product_Category_1`, `Product_Category_2`, etc
        """
        if not self._product_category_encoder:
            self._product_category_encoder = ProductCategoriesEncoder(encoding=encoding, product_category_cols=self.product_category_cols)
            self._product_category_encoder.fit(df)

        df = self._product_category_encoder.transform(df)

        return df
