import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


FILEPATH_TRAIN = './data/training.txt'
FILEPATH_TEST = './data/test.txt'
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
          'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'target']
Z_THRESHOLD = 3
min_max_scaler = preprocessing.MinMaxScaler()

def get_data():
    """Return two pandas dataframes: training data and test data."""
    train = pd.read_csv(FILEPATH_TRAIN, sep='\t', names=HEADER)
    test = pd.read_csv(FILEPATH_TEST, sep='\t', names=HEADER[:-1])
    return train, test


def native_country_corrector(x):
    x = x.strip()
    return x if x == 'United-States' else 'Other'


def preprocess_data(data, train):
    """Perform preprocessing operations on data. For normalization, check if data is for training or test.

    Parameters:
        data: dataframe to preprocess.
        train: boolean, True if 'data' is for training.

    Returns:
        preprocessed data.
    """

    data = data.drop(['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss'], axis=1)
    data_num = data.select_dtypes(include=np.number)

    # normalize numerical data.
    np_scaled = min_max_scaler.fit_transform(data_num) if train else min_max_scaler.transform(data_num)
    data_num_normalized = pd.DataFrame(np_scaled, columns=data_num.columns)

    # replace normalized columns in the dataframe.
    for col in data_num_normalized.columns:
        data[col] = data_num_normalized[col]

    columns_cat = []
    for c in data.columns:
        if c not in data_num.columns:
            columns_cat.append(c)

    # remove rows with null categorical values.
    if train:
        data = data[(data['workclass'] != " ?") & (data['occupation'] != " ?") & (data['native_country'] != " ?")]

    column_native = data['native_country']
    data['native_country'] = list(map(native_country_corrector, column_native))
    data = pd.get_dummies(data)

    # in test set I cannot remove rows. OHE creates columns related to null values.
    if not train:
        data = data.drop(['workclass_ ?', 'occupation_ ?'], axis=1)

    return data


def train_validation_split(train, percentage):
    """Split training data in training and validation sets.

    Parameters:
        train: training data to split.
        percentage: float < 1 representing the percentage of 'data' in the validation set.

    Returns:
        X_train.
        X_valid.
        y_train.
        y_valid.
    """

    # to address the imbalance problem
    target_high = train[train['target_ >50K'] == 1]
    for i in range(2):
        train = train.append(target_high)

    train = train.to_numpy()
    x_train_ov = train[:, :-2]
    y_train_ov = train[:, -2]

    X_train, X_valid, y_train, y_valid = train_test_split(x_train_ov, y_train_ov, test_size=percentage, random_state=42)
    return X_train, X_valid, y_train, y_valid

