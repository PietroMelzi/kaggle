import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


FILEPATH_TRAIN = './data/training.txt'
FILEPATH_TEST = './data/test.txt'
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
          'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'target']


def get_data():
    """Return three numpy arrays: training data, target for training, test data."""
    train = pd.read_csv(FILEPATH_TRAIN, sep='\t', index=False, header=HEADER)
    test = pd.read_csv(FILEPATH_TEST, sep='\t')
    train = np.array(train.values)
    test = np.array(test.values)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    return x_train, y_train, test


def analyze_data():
    train = pd.read_csv(FILEPATH_TRAIN, sep='\t', names=HEADER)
    train_temp = train.loc[(train['capital_gain'] == 0) & (train['capital_loss'] == 0)]
    print("Percentage of data with capital gain/loss equal to 0: {}%".format(100 * len(train_temp) / len(train)))

    # some information on numerical attributes.
    train_num = train.select_dtypes(include=np.number)
    for c in train_num.columns:
        train_num_col = train_num[c].to_numpy()
        print("{}: mean = {}, std = {}, min = {}, max = {}.".format(c, train_num_col.mean(), train_num_col.std(),
                                                                    train_num_col.min(), train_num_col.max()))
    # some information on categorical attributes.
    for c in train.columns:
        if c not in train_num.columns:
            train_cat_col = train[c].values
            print("{} - {}.".format(c, set(train_cat_col)))
    #         for t in train_cat_col:
    #             if (t == " ?"):
    #                 print("here")
    #         print("{}: null values = {}".format(c, len([t for t in train_cat_col if t == "?"])))
    # train = pd.get_dummies(train)
    # print("here")



# x_train, y_train, x_test = get_data()
analyze_data()



