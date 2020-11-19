import os
import json
import torch
from tqdm import tqdm

# import user-written files
import data.loader as loader
from utils.utils import compute_normalizer
# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState
import utils.datavisualizer as dv

if __name__ == "__main__":
    # set (high level) options dictionary
    main_options = {
        'model_path': os.getcwd() + '/log/',
        'folder_names': 'model_11_11_full_',
        't_not_trained': 0,  # number of last data days not used for training
        'prediction_steps': 14,  # number of recursive prediction steps
        'data_file_name': os.getcwd() + '/data/covid/covid_uu_11_11.csv',
    }

    # allocation
    y_recursive_all = []

    tqdm.write("Loop over all folders and make predictions with each model.")
    # loop over all folders
    folders = [(i) for i in os.listdir(main_options['model_path']) if i.startswith(main_options['folder_names'])]
    for i, dir in enumerate(folders):
        tqdm.write("\tPerform predictions with model {}.".format(i + 1))

        dir = os.path.join(main_options['model_path'], dir)
        # load options
        with open(os.path.join(dir, "options.txt")) as f:
            opt = json.loads(f.read())

        # get correct computing device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # overwrite dataset path to be sure to take the correct one (can change over different machines)
        opt["data_file_name"] = main_options["data_file_name"]

        # get the options
        opt['device'] = device
        opt['dataset_options'] = dynsys_params.get_dataset_options(opt['dataset'])
        opt['model_options'] = model_params.get_model_options(opt['model'], opt['dataset'],
                                                              opt['dataset_options'])
        opt['train_options'] = train_params.get_train_options(opt['dataset'])
        opt['test_options'] = train_params.get_test_options()
        # set model options to user chosen
        opt['model_options'].h_dim = opt['user_model_opts']['h_dim']
        opt['model_options'].z_dim = opt['user_model_opts']['z_dim']
        opt['model_options'].n_layers = opt['user_model_opts']['n_layers']

        # Specifying datasets
        kwargs = {'data_file_name': opt['data_file_name'], 't_not_trained': opt['t_not_trained']}
        loaders = loader.load_dataset(dataset=opt["dataset"],
                                      dataset_options=opt["dataset_options"],
                                      train_batch_size=opt["train_options"].batch_size,
                                      test_batch_size=opt["test_options"].batch_size,
                                      **kwargs)

        # Compute normalizers
        if opt["normalize"]:
            normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
        else:
            normalizer_input = normalizer_output = None

        # Define model
        modelstate = ModelState(seed=opt["seed"],
                                nu=loaders["train"].nu, ny=loaders["train"].ny,
                                model=opt["model"],
                                options=opt,
                                normalizer_input=normalizer_input,
                                normalizer_output=normalizer_output)
        modelstate.model.to(opt['device'])

        # load model
        path = os.path.join(dir, 'model')
        file_name = opt['dataset'] + '_bestModel.ckpt'
        modelstate.load_model(path, file_name)
        modelstate.model.to(opt['device'])
        modelstate.model.eval()

        # %% do recursive prediction
        for (u_test, y_test) in loaders['test']:
            # get time stamps
            T = u_test.size(2) - main_options['t_not_trained']
            N = main_options['prediction_steps']
            # getting output distribution parameter only implemented for selected models
            u_test = u_test.to(opt['device'])
            y_recursive = modelstate.model.generate_recursively(u_test, T, N)
            # convert to cpu and to numpy for evaluation
            y_recursive = y_recursive.cpu().squeeze()
            y_test = y_test.cpu().detach().numpy()
        # append predictions
        y_recursive_all.append(y_recursive)
    tqdm.write("Predictions done!")

    # make y_recursive tensor
    y_recursive = torch.stack(y_recursive_all).detach().numpy()
    # get statistics
    y_recursive_mu = y_recursive.mean(axis=0)
    y_recursive_std = y_recursive.std(axis=0)

    # %% plot resulting predictions
    label_y = ['true', '1-step ahead prediction']
    title = ['pat', 'pat_icu', 'in', 'out', 'pos']
    dv.plot_time_sequence_uncertainty(y_test.squeeze(),
                                      y_recursive_mu,
                                      y_recursive_std,
                                      label_y,
                                      title,
                                      opt,
                                      path=main_options['model_path'],
                                      file_name='ensemble_predictions_1StepAhead_')
