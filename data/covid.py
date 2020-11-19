import numpy as np
import pandas as pd
from data.base import IODataset


def create_covid_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    # which data set to use
    if 'data_file_name' in kwargs:
        file_name_data = kwargs['data_file_name']
    else:
        raise Exception("Need file input name!")
    if 't_not_trained' in kwargs:
        t_not_trained = kwargs['t_not_trained']
    else:
        t_not_trained = 0

    # read the file into variable
    df = pd.read_csv(file_name_data)
    # remove date column
    data = df.to_numpy()[:, 1:]

    # input data u is [pat(t-1), pat_icu(t-1), pat_in(t-1), pat_out(t-1) pos(t-1)]
    # output data y is [pat(t), pat_icu(t)]
    # input an output are np arrays of size (len x 5) for input and (len x 2) for output
    input = data[:-1]
    output = data[1:]

    # make last n days optional for training. Validation and test data is the full dataset
    n = t_not_trained
    if n == 0:
        u_train, y_train = input, output
    else:
        u_train = input[:-n]
        y_train = output[:-n]
    u_test = u_val = input
    y_test = y_val = output

    # get correct dimensions
    u_train = u_train[..., None]
    y_train = y_train[..., None]
    u_val = u_val[..., None]
    y_val = y_val[..., None]
    u_test = u_test[..., None]
    y_test = y_test[..., None]

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test
