import argparse
import torch


def get_train_options(dataset_name):
    train_parser = argparse.ArgumentParser(description='training parameter')
    train_parser.add_argument('--clip', type=int, default=10, help='clipping of gradients')
    train_parser.add_argument('--lr_scheduler_nstart', type=int, default=10, help='learning rate scheduler start epoch')
    train_parser.add_argument('--test_every', type=int, default=5, help='test during training after every n epoch')

    if dataset_name == 'covid':
        train_parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
    else:
        raise Exception("Unimplemented dataset!")

    # change batch size to higher value if trained on cuda device
    if torch.cuda.is_available():
        train_parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    else:
        train_parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    train_options = train_parser.parse_args()

    return train_options


def get_test_options():
    test_parser = argparse.ArgumentParser(description='testing parameter')
    test_parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # 128
    test_options = test_parser.parse_args()

    return test_options
