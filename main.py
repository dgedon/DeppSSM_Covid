# import generic libraries
import torch.utils.data
import pandas as pd
import os
import torch
from tqdm import tqdm

# import user-written files
import data.loader as loader
import training
import testing
from utils.utils import compute_normalizer
from utils.utils import save_options
# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_single(_options, _path_log, _file_name):
    tqdm.write('Run file: main_single.py')

    # get correct computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write('Device: {}'.format(device))

    # get the options
    _options['device'] = device
    _options['dataset_options'] = dynsys_params.get_dataset_options(_options['dataset'])
    _options['model_options'] = model_params.get_model_options(_options['model'], _options['dataset'],
                                                               _options['dataset_options'])
    _options['train_options'] = train_params.get_train_options(_options['dataset'])
    _options['test_options'] = train_params.get_test_options()
    # set model options to user chosen
    _options['model_options'].h_dim = _options['user_model_opts']['h_dim']
    _options['model_options'].z_dim = _options['user_model_opts']['z_dim']
    _options['model_options'].n_layers = _options['user_model_opts']['n_layers']

    # print model type and dynamic system type
    tqdm.write('\n\tModel Type: {}'.format(_options['model']))
    tqdm.write('\tDynamic System: {}\n'.format(_options['dataset']))

    # Specifying datasets
    kwargs = {'data_file_name': _options['data_file_name'], 't_not_trained': _options['t_not_trained']}
    loaders = loader.load_dataset(dataset=_options["dataset"],
                                  dataset_options=_options["dataset_options"],
                                  train_batch_size=_options["train_options"].batch_size,
                                  test_batch_size=_options["test_options"].batch_size,
                                  **kwargs)

    # Compute normalizers
    if _options["normalize"]:
        normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
    else:
        normalizer_input = normalizer_output = None

    # Define model
    modelstate = ModelState(seed=_options["seed"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            model=_options["model"],
                            options=_options,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output)
    modelstate.model.to(_options['device'])

    # save the options
    save_options(_options, _path_log, 'options.txt')

    # allocation
    df = {}
    if _options['do_train']:
        # train the model
        df = training.run_train(modelstate=modelstate,
                                loader_train=loaders['train'],
                                loader_valid=loaders['valid'],
                                options=_options,
                                dataframe=df,
                                path_general=_path_log,
                                file_name_general=_file_name)

    if _options['do_test']:
        # test the model
        df = testing.run_test(_options, loaders, df, _path_log, _file_name)

    # save data
    # get saving path
    path = _path_log + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # to pandas
    df = pd.DataFrame(df)
    # filename
    file_name = _file_name + '.csv'
    # save data
    df.to_csv(path + file_name)


# %%
if __name__ == "__main__":
    # set (high level) options dictionary
    options = {
        'dataset': 'covid',
        'model': 'STORN',
        'do_train': False,
        'do_test': True,
        'normalize': True,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': True,
        'savefig': True,
        'test_name': 'model_11_11_full_1',
        'data_file_name': os.getcwd() + '/data/covid/covid_uu_11_11.csv',
        't_not_trained': 0,  # number of last data days not used for training
        'prediction_steps': 14,  # number of recursive prediction steps
        'user_model_opts': {
            'h_dim': 300,
            'z_dim': 10,
            'n_layers': 3},
    }

    # get saving path
    path_log = os.getcwd() + '/log/{}/'.format(options['test_name'])
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    # get saving file names
    file_name = options['dataset']

    run_main_single(options, path_log, file_name)
