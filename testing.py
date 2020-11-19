# import generic libraries
import torch
import matplotlib.pyplot as plt
# import user-writte files
import utils.datavisualizer as dv
import utils.dataevaluater as de
from utils.utils import get_n_params
from models.model_state import ModelState
from utils.utils import compute_normalizer


def run_test(options, loaders, df, path_general, file_name_general, **kwargs):
    # switch to cpu computations for testing
    # options['device'] = 'cpu'

    # Compute normalizers (here just used for initialization, real values loaded below)
    if options["normalize"]:
        normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
    else:
        normalizer_input = normalizer_output = None

    # Define model
    modelstate = ModelState(seed=options["seed"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            model=options["model"],
                            options=options,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output)
    modelstate.model.to(options['device'])

    # load model
    path = path_general + 'model/'
    file_name = file_name_general + '_bestModel.ckpt'
    modelstate.load_model(path, file_name)
    modelstate.model.to(options['device'])
    modelstate.model.eval()

    # %% plot and save the loss curve
    dv.plot_losscurve(df, options, path_general, file_name_general)

    # %% do 1 step ahead prediction.
    for (u_test, y_test) in loaders['test']:
        # getting output distribution parameter only implemented for selected models
        u_test = u_test.to(options['device'])
        _, y_stepahead, _ = modelstate.model.generate(u_test)
        # convert to cpu and to numpy for evaluation
        y_stepahead = y_stepahead.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()

    # %% plot resulting predictions
    label_y = ['true', '1-step ahead prediction']
    title = ['pat', 'pat_icu', 'in', 'out', 'pos']
    dv.plot_time_sequence(y_test,
                          y_stepahead,
                          label_y,
                          title,
                          options,
                          path_general=path_general,
                          file_name_general=file_name_general + '_1StepAhead_')

    # %% do recursive prediction
    """
    1-step ahead prediction until time T=(max_length-t_not_trained) to obtain the correct hidden states.
    Then do recursive predictions with output of t as input for t+1
    Do this recursive prediction for N steps.
    Bootstrap this X times with random start to get statistics.
    """
    for (u_test, y_test) in loaders['test']:
        # get time stamps
        T = u_test.size(2) - options['t_not_trained']
        N = options['prediction_steps']
        # getting output distribution parameter only implemented for selected models
        u_test = u_test.to(options['device'])
        y_recursive = modelstate.model.generate_recursively(u_test, T, N)
        # convert to cpu and to numpy for evaluation
        y_recursive = y_recursive.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()

    # %% plot resulting predictions
    label_y = ['true', 'recrusive prediction']
    title = ['pat', 'pat_icu', 'in', 'out', 'pos']
    dv.plot_time_sequence(y_test,
                          y_recursive,
                          label_y,
                          title,
                          options,
                          path_general=path_general,
                          file_name_general=file_name_general + '_recursive')

    # %% compute performance values
    # compute RMSE
    rmse = de.compute_rmse(y_test, y_stepahead, doprint=True)

    # %% Collect data
    # options_dict
    options_dict = {'h_dim': options['model_options'].h_dim,
                    'z_dim': options['model_options'].z_dim,
                    'n_layers': options['model_options'].n_layers,
                    'seq_len_train': options['dataset_options'].seq_len_train,
                    'batch_size': options['train_options'].batch_size,
                    'lr_scheduler_nepochs': options['train_options'].lr_scheduler_nepochs,
                    'lr_scheduler_factor': options['train_options'].lr_scheduler_factor
                    }
    # test_dict
    test_dict = {'rmse': rmse}
    # dataframe
    df.update(options_dict)
    df.update(test_dict)

    return df
