import matplotlib.pyplot as plt
import torch
import numpy as np
import os


# %% plots the resulting time sequence
def plot_time_sequence(data_y_true, data_y_sample, label_y, title, options, path_general, file_name_general,
                       batch_show=0):
    # storage path
    file_name = file_name_general + '_timeEval'
    path = path_general + 'timeEval/'

    # get number of outputs
    num_outputs = data_y_sample.shape[1]

    # get number of columns
    num_cols = 1

    # initialize figure
    plt.figure(1, figsize=(5 * num_cols, 5 * num_outputs), dpi=300)

    # plot outputs
    for j in range(0, num_outputs):
        # output yk
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))
        # plot true data
        plt.plot(data_y_true[batch_show, j, :].squeeze(), label='{}'.format(label_y[0]))

        # plot samples mu
        length = len(data_y_sample[batch_show, j, :])
        x = np.linspace(0, length - 1, length)
        y = data_y_sample[batch_show, j, :].squeeze()
        plt.plot(x, y, label='{}'.format(label_y[1]))

        # plot vertical line for training/validation split
        x0 = len(data_y_true[batch_show, j, :]) - options['t_not_trained']
        plt.vlines(x0, 0, max(data_y_true[batch_show, j, :]), label='training split')

        # plot settings
        plt.title('Output: "{}"'.format(title[j]))
        plt.ylabel('{}'.format(title[j]))
        plt.xlabel('time steps $k$')
        plt.legend()

    # save figure
    if options['savefig']:
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()
    plt.close(1)


# %% plots the resulting time sequence
def plot_time_sequence_uncertainty(data_y_true, data_y_sample_mu, data_y_sample_std, label_y, title, options,
                                   path, file_name):
    # get number of outputs
    num_outputs = data_y_sample_mu.shape[0]

    # get number of columns
    num_cols = 1

    # initialize figure
    plt.figure(1, figsize=(5 * num_cols, 5 * num_outputs), dpi=300)

    # plot outputs
    for j in range(0, num_outputs):
        # output yk
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))
        # plot true data
        plt.plot(data_y_true[j, :].squeeze(), label='{}'.format(label_y[0]))

        # plot samples mu
        length = len(data_y_sample_mu[j, :])
        x = np.linspace(0, length - 1, length)
        y_mu = data_y_sample_mu[j, :].squeeze()
        y_std = data_y_sample_std[j, :].squeeze()
        plt.plot(x, y_mu, label='{}'.format(label_y[1]))
        plt.fill_between(x, y_mu, y_mu + 2 * y_std, alpha=0.3, facecolor='r')
        plt.fill_between(x, y_mu, y_mu - 2 * y_std, alpha=0.3, facecolor='r')

        # plot vertical line for training/validation split
        x0 = len(data_y_true[j, :]) - options['t_not_trained']
        plt.vlines(x0, 0, max(data_y_true[j, :]), label='latest data')

        # plot settings
        plt.title('Output: "{}"'.format(title[j]))
        plt.ylabel('{}'.format(title[j]))
        plt.xlabel('time steps $k$')
        plt.legend()

    # save figure
    if options['savefig']:
        # storage path
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()
    plt.close(1)


# %% plot and save the loss curve
def plot_losscurve(df, options, path_general, file_name_general, removedata=True):
    # only if df has values
    if 'all_losses' in df:
        # storage path
        file_name = file_name_general + '_loss'
        path = path_general + '/loss/'

        # get data to plot loss curve
        all_losses = df['all_losses']
        all_vlosses = df['all_vlosses']

        # plot loss curve
        plt.figure(1, figsize=(5, 5))
        xval = np.linspace(0, options['train_options'].test_every * (len(all_losses) - 1), len(all_losses))
        plt.plot(xval, all_losses, label='Training set')
        plt.plot(xval, all_vlosses, label='Validation set')  # loss_test_store_idx,
        plt.xlabel('Number Epochs ')
        plt.ylabel('Loss')
        plt.title('Loss of {} with (h,z,n)=({},{},{})'.format(options['dataset'],
                                                              options['model_options'].h_dim,
                                                              options['model_options'].z_dim,
                                                              options['model_options'].n_layers))
        plt.legend()
        plt.yscale('log')
        # save model
        if options['savefig']:
            # check if path exists and create otherwise
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + file_name + '.png', format='png')
        # show the model
        if options['showfig']:
            plt.show()
        plt.close(1)

        # delete loss value matrices from dictionary
        if removedata:
            del df['all_losses']
            del df['all_vlosses']

    return df
