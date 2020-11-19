from data.base import DataLoaderExt
from data.covid import create_covid_datasets


def load_dataset(dataset, dataset_options, train_batch_size, test_batch_size, **kwargs):
    if dataset == 'covid':
        dataset_train, dataset_valid, dataset_test = create_covid_datasets(dataset_options.seq_len_train,
                                                                           dataset_options.seq_len_val,
                                                                           dataset_options.seq_len_test,
                                                                           **kwargs)
    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    # Dataloader
    loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
    loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
    loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
