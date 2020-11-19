import argparse


def get_dataset_options(dataset_name):

    if dataset_name == 'covid':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: covid')
        dataset_parser.add_argument('--y_dim', type=int, default=5, help='dimension of output')
        dataset_parser.add_argument('--u_dim', type=int, default=5, help='dimension of input')
        dataset_parser.add_argument('--seq_len_train', type=int, default=None, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type =int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=None, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()
    else:
        raise Exception("Unimplemented dataset!")

    return dataset_options
