# coding:utf-8
from .cifar10 import mycifar10
# from .channel3_data import channel3_closs
from .channel3_data_b import channel3_closs_b
from .yamaha import yamaha_dataset


def load_dataset(root, dataset_name, seed, height, width, approach, data_path, normal_class, known_outlier_class):
    """Loads the dataset."""

    implemented_datasets = ('yamaha', 'mnist', 'cifar10', 'No.6', 'No.21', 'No.23', 'No.24', 'No.26')
    assert dataset_name in implemented_datasets

    dataset = None

    '''
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)
    '''

    if dataset_name == 'cifar10':
        train_dataset, val_dataset, test_dataset, classes = mycifar10(root=data_path, height=height, width=width, approach=approach, 
                                                                        normal_class=normal_class, known_outlier_class=known_outlier_class)

    if (dataset_name == 'No.6') | (dataset_name == 'No.21') | (dataset_name == 'No.23') | (dataset_name == 'No.24') | (dataset_name == 'No.26'):
        train_dataset, val_dataset, test_dataset, classes = channel3_closs_b(root=data_path, seed=seed, height=height, width=width, approach=approach,
                                                                        normal_class=normal_class, known_outlier_class=known_outlier_class)
    
    if dataset_name == "yamaha":
        train_dataset, val_dataset, test_dataset, classes = yamaha_dataset(root=data_path, seed=seed, height=height, width=width, approach=approach,
                                                                        normal_class=normal_class, known_outlier_class=known_outlier_class)

    return train_dataset, val_dataset, test_dataset, classes