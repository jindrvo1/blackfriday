from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from google.cloud import storage

from enum import Enum


class EncodingType(Enum):
    BINARY = 0
    ORDINAL = 1
    ONE_HOT = 2


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


    def __init__(self, path: str, test_path: str = None):
        if path.startswith('gs://'):
            self.gcs_client = storage.Client.create_anonymous_client()

            self.df = pd.read_csv(path)
            self.df_test = pd.read_csv(test_path) if test_path else None
        else:
            self.df = self._load_data_file(path)
            self.df_test = self._load_data_file(test_path) if test_path else None


    def preprocess_dfs(self, encodings: dict[str, EncodingType], return_res: bool = True) -> None | tuple[pd.DataFrame, pd.DataFrame | None]:
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
        test_size: float = 0.2,
        shuffle: bool = True,
        target_col: str = 'Purchase',
        cols_to_drop: list[str] = ['User_ID', 'Product_ID'],
        return_res: bool = True,
    ) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
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
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        else:
            self.X_train = X.sample(frac=1) if shuffle else X
            self.y_train = y.sample(frac=1) if shuffle else y
            self.X_val, self.y_val = None, None

        self.X_test = self.df_test_encoded.drop(columns=cols_to_drop) if self.df_test_encoded is not None else None

        return (self.X_train, self.y_train, self.X_val, self.y_val, self.X_test) if return_res else None


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
        """Encodes the `Age` column of the `df` dataframe to respective
        ordinal integer values from 0 to N where N is the number of
        different age categories in the `df` dataframe

        Args:
            df (pd.DataFrame): The dataframe with the 'Age' column populated
                with age categories represented by string values

        Returns:
            pd.DataFrame: The dataframe with the 'Age' column's values mapped
                to ordinal integer values
        """
        if not self._age_encoder:
            self._age_encoder = self._prepare_col_encoder(df, encoding, 'Age')

        return self._encode_col(df, self._age_encoder, 'Age')


    def _process_city_category_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """One-hot encodes the `City_Category` column of the `df` dataframe.
        The original `City_Category` column is dropped in favour of its encoded
        representation in N-1 new columns where N is the number of different city
        categories. The first column of the encoded representation is dropped as
        the vector [1, 0, 0, ...] can be uniquely identified by a vector of all zeros

        Args:
            df (pd.DataFrame): The dataframe with the 'City_Category' column populated
                with city categories represented by string values

        Returns:
            pd.DataFrame: The dataframe with the 'City_Category' column replaced
                by N-1 columns representing its one-hot encoded representation.
                The output column names are `City_Category_1`, `City_Category_2`, etc
        """
        if not self._city_category_encoder:
            self._city_category_encoder = self._prepare_col_encoder(df, encoding, 'City_Category')

        return self._encode_col(df, self._city_category_encoder, 'City_Category')


    def _process_stay_in_current_city_years_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """Encodes the `Stay_In_Current_City_Years` column of the `df` dataframe to
        respective ordinal integer values from 0 to N where N is the number of
        different categories in the column

        Args:
            df (pd.DataFrame): The dataframe with the 'Stay_In_Current_City_Years'
                column populated with string values

        Returns:
            pd.DataFrame: The dataframe with the 'Stay_In_Current_City_Years' column's
                values mapped to ordinal integer values
        """
        if not self._stay_in_current_city_years_encoder:
            self._stay_in_current_city_years_encoder = self._prepare_col_encoder(df, encoding, 'Stay_In_Current_City_Years')

        return self._encode_col(df, self._stay_in_current_city_years_encoder, 'Stay_In_Current_City_Years')

    def _process_occupation_col(self, df: pd.DataFrame, encoding: EncodingType) -> pd.DataFrame:
        """One-hot encodes the `Occupation` column of the `df` dataframe.
        The original `Occupation` column is dropped in favour of its encoded
        representation in N-1 new columns where N is the number of different occupation
        categories. The first column of the encoded representation is dropped as
        the vector [1, 0, 0, ...] can be uniquely identified by a vector of all zeros

        Args:
            df (pd.DataFrame): The dataframe with the 'Occupation' column populated
                with occupation categories represented by integer values

        Returns:
            pd.DataFrame: The dataframe with the 'Occupation' column replaced
                by N-1 columns representing its one-hot encoded representation.
                The output column names are `Occupation_1`, `Occupation_2`, etc
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
