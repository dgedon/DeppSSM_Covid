import argparse


def get_model_options(model_type, dataset_name, dataset_options):

    y_dim = dataset_options.y_dim
    u_dim = dataset_options.u_dim

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--y_dim', type=int, default=y_dim, help='dimension of output')
    model_parser.add_argument('--u_dim', type=int, default=u_dim, help='dimension of input')

    if dataset_name == 'covid':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=3, help='dimension of stochastic latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')
    else:
        raise Exception("Unimplemented dataset!")

    model_options = model_parser.parse_args()

    return model_options
