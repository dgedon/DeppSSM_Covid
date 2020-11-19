import torch
import torch.utils
import torch.utils.data
import numpy as np
import time
from tqdm import tqdm


def run_train(modelstate, loader_train, loader_valid, options, dataframe, path_general, file_name_general):
    def validate(loader):
        modelstate.model.eval()
        total_vloss = 0
        total_batches = 0
        total_points = 0

        for (u, y) in loader:
            # data to device
            u = u.to(options['device'])
            y = y.to(options['device'])

            # forward pass over model
            with torch.no_grad():
                vloss_ = modelstate.model(u, y)

            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_vloss += vloss_.item()

        return total_vloss / total_points

    def train(epoch):
        # model in training mode
        modelstate.model.train()
        # initialization
        total_loss = 0
        total_batches = 0
        total_points = 0

        for (u, y) in loader_train:
            # current batch_size
            bs = u.size(0)

            # data to device
            u = u.to(options['device'])
            y = y.to(options['device'])

            # set the optimizer
            modelstate.optimizer.zero_grad()
            # forward pass over model
            loss_ = modelstate.model(u, y)
            # NN optimization
            loss_.backward()
            modelstate.optimizer.step()

            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_loss += loss_.item()

        return total_loss / total_points

    try:
        model_options = options['model_options']
        train_options = options['train_options']

        modelstate.model.train()
        # Train
        all_losses = []
        all_vlosses = []
        best_vloss = np.inf

        # Extract initial learning rate
        lr = train_options.init_lr

        # output parameter
        best_epoch = 0

        # initialise progress bar
        process_desc = "Train-loss: {:2.3e}; Valid-loss: {:2.3e}; LR: {:2.3e}"
        progress_bar = tqdm(initial=0, leave=True, total=train_options.n_epochs,
                            desc=process_desc.format(0, 0, 0), position=0)

        for epoch in range(train_options.n_epochs):
            # Train and validate
            loss = train(epoch)
            # validate every n epochs
            if epoch % train_options.test_every == 0:
                vloss = validate(loader_valid)
                # Save losses
                all_losses += [loss]
                all_vlosses += [vloss]

                if vloss < best_vloss:  # epoch == train_options.n_epochs:  #
                    best_vloss = vloss
                    # save model
                    path = path_general + 'model/'
                    file_name = file_name_general + '_bestModel.ckpt'
                    modelstate.save_model(epoch, vloss, path, file_name)
                    # torch.save(model.state_dict(), path + file_name)
                    best_epoch = epoch

                # lr scheduler
                if epoch >= train_options.lr_scheduler_nstart:
                    # if the validation loss does not decrease for n_epochs, then decrease learning rate
                    if len(all_vlosses) > train_options.lr_scheduler_nepochs and \
                            vloss >= max(all_vlosses[int(-train_options.lr_scheduler_nepochs - 1):-1]):
                        # reduce learning rate
                        lr = lr / train_options.lr_scheduler_factor
                        # adapt new learning rate in the optimizer
                        for param_group in modelstate.optimizer.param_groups:
                            param_group['lr'] = lr
                        message = 'Learning rate adapted in epoch {} with valid loss {:2.6e}. New learning rate {:.3e}.'
                        tqdm.write(message.format(epoch, vloss, lr))
            # Update train bar
            progress_bar.desc = process_desc.format(loss, vloss, lr)
            progress_bar.update(1)

            # Early stopping condition
            if lr < train_options.min_lr:
                break
        progress_bar.close()

    except KeyboardInterrupt:
        tqdm.write('\n')
        tqdm.write('-' * 89)
        tqdm.write('Exiting from training early')
        tqdm.write('-' * 89)

    # save data in dictionary
    train_dict = {'all_losses': all_losses,
                  'all_vlosses': all_vlosses,
                  'best_epoch': best_epoch,
                  'total_epoch': epoch}
    # overall options
    dataframe.update(train_dict)

    return dataframe
