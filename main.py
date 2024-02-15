# coding:utf-8
import os
import click
import time
import logging
import random
import numpy as np
import torch
from pretrain import AETrainer
from trainer import Trainer
from datasets.main import load_dataset
from utils.plot import plot_images_grid, plot_hist, plot_all_hist

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['yamaha', 'mnist', 'cifar10', 'No.6', 'No.21', 'No.23', 'No.24', 'No.26']))
@click.argument('save_path', type=click.Path(exists=False))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--height', type=int, default=256)
@click.option('--width', type=int, default=256)
@click.option('--approach', type=click.Choice(['DSVDD', 'DSAD']), default='DSVDD')
@click.option('--rep_dim', type=int, default=32)
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--nu', type=float, default=1.0, help='hyperparameter for getting radius (must be 0 < nu < 1.0).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=1, help='Set seed.')
#@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
#              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--lr_milestone', type=str, default='150', multiple=False,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=32, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=False,
              help='Pretrain neural network parameters via autoencoder.')
#@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
#              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_lr_milestone', type=str, default='80', multiple=False,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=32, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')

def main(dataset_name, save_path, data_path, height, width, approach, rep_dim, eta, nu, device, seed, n_epochs, lr, lr_milestone,
           batch_size, weight_decay, pretrain, ae_n_epochs, ae_lr, ae_lr_milestone, ae_batch_size, ae_weight_decay,normal_class, known_outlier_class):
    start_time = time.time()
    # スケジューラーをstrからintに変換
    lr_milestone = lr_milestone.split(',')
    lr_milestone = [int(str) for str in lr_milestone]
    ae_lr_milestone = ae_lr_milestone.split(',')
    ae_lr_milestone = [int(str) for str in ae_lr_milestone]
    if (pretrain):
        save_path = save_path + ('pretrain_true/h=%d_w=%d_dim=%d_eta=%g_nu=%g_seed=%d_epoch=%d;%d_lr=%g;%g_scheduler=%s;%s_batch=%d;%d_weightdecay=%g;%g_normal=%d_outlier=%d/' % (height, width, rep_dim, eta, nu, seed, n_epochs, ae_n_epochs, lr, ae_lr, lr_milestone, ae_lr_milestone, batch_size, ae_batch_size, weight_decay, ae_weight_decay, normal_class, known_outlier_class))
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = save_path + ('pretrain_False/h=%d_w=%d_dim=%d_eta=%g_nu=%g_seed=%d_epoch=%d_lr=%g_scheduler=%s_batch=%d_weightdecay=%g_normal=%d_outlier=%d/' % (height, width, rep_dim, eta, nu, seed, n_epochs, lr, lr_milestone, batch_size, weight_decay, normal_class, known_outlier_class))
        os.makedirs(save_path, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = save_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # データ読み込み
    train_dataset, val_dataset, test_dataset, classes = load_dataset(root=data_path, dataset_name=dataset_name, seed=seed, height=height, width=width, approach=approach, data_path=data_path,
                                                                      normal_class=normal_class, known_outlier_class=known_outlier_class)
    t = 2 # 交差検証法(3分割)のデータ選択(0 or 1 or 2)
    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Known outlier class: %d' % known_outlier_class)
    logger.info('Train dataset size: %d' % len(train_dataset[t]))
    logger.info('Validation dataset size: %d' % len(val_dataset[t]))
    logger.info('Test dataset size: %d' % len(test_dataset))
    logger.info('optimizer: Adam')
    logger.info('eta: %g' % eta)
    logger.info('nu: %g' % nu)
    logger.info('Set seed to %d' % seed)
    if (pretrain):
        logger.info('------ pretrain setting ------')
        logger.info('ae_learning rate: %g' % ae_lr)
        logger.info('ae_epochs: %d' % ae_n_epochs)
        logger.info('ae_learning rate scheduler milestones: %s' % ae_lr_milestone)
        logger.info('ae_batch size: %d' % ae_batch_size)
        logger.info('ae_weight decay: %g' % ae_weight_decay)
    logger.info('------ train setting ------')
    logger.info('learning rate: %g' % lr)
    logger.info('epochs: %d' % n_epochs)
    logger.info('learning rate scheduler milestones: %s' % lr_milestone)
    logger.info('batch size: %d' % batch_size)
    logger.info('weight decay: %g' % weight_decay)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device: %s' % device)

    # pretraining
    if (pretrain):
        aeTrainer = AETrainer(dataset_name=dataset_name, save_path=save_path, device=device, dim=64, rep_dim=rep_dim, height=height, width=width,
                                n_epochs=ae_n_epochs, lr=ae_lr, lr_milestone=ae_lr_milestone, batch_size=ae_batch_size, weight_decay=ae_weight_decay)
        aeTrainer.train(train_dataset=train_dataset, val_dataset=val_dataset, classes=classes)
        aeTrainer.test(test_dataset=test_dataset, classes=classes)

    # training
    trainer = Trainer(dataset_name=dataset_name, save_path=save_path, device=device, dim=64, rep_dim=rep_dim, height=height, width=width,
                      	n_epochs=n_epochs, lr=lr, lr_milestone=lr_milestone, batch_size=batch_size, weight_decay=ae_weight_decay,
                      	normal_class=normal_class, known_outlier_class=known_outlier_class)
    center = trainer.train(train_dataset=train_dataset, val_dataset=val_dataset, classes=classes, pretrain=pretrain, nu=nu, eta=eta)
    idx, labels, scores, r, c = trainer.test(test_dataset=test_dataset, classes=classes, c=center)

    torch.save(c, os.path.join(save_path, 'center.pth'))

    # 画像データと異常スコアを対応させながらソート
    idx_all_sorted = idx[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = idx[labels == 0][np.argsort(scores[labels == 0])]
    idx_outlier_sorted = idx[labels == 1][np.argsort(scores[labels == 1])]
    #idx_gray_sorted = idx[labels == 2][np.argsort(scores[labels == 2])] #

    score_all_sorted = scores[np.argsort(scores)]
    score_nomal_sorted = scores[labels == 0][np.argsort(scores[labels == 0])]
    score_outlier_sorted = scores[labels == 1][np.argsort(scores[labels == 1])]
    #score_gray_sorted = scores[labels == 2][np.argsort(scores[labels == 2])] #

    label_all_sorted = labels[idx_all_sorted]
    label_normal_sorted = labels[idx_normal_sorted]
    label_outlier_sorted = labels[idx_outlier_sorted]
    #label_gray_sorted = labels[idx_gray_sorted] #

    if (dataset_name != 'mnist'):
        t_data = np.array([td[0].to('cpu').detach().numpy().copy() for td in test_dataset])
        t_data = np.transpose(t_data.data, (0,2,3,1))

        X_all_low = t_data[idx_all_sorted[:40]]
        X_all_high = t_data[idx_all_sorted[-40:]]
        X_normal_low = t_data[idx_normal_sorted[:40]]
        X_normal_high = t_data[idx_normal_sorted[-40:]]
        X_outlier_high = t_data[idx_outlier_sorted[-40:]]
        #X_gray_high = t_data[idx_gray_sorted[-50:]] #

        X_all_low_score = torch.tensor(score_all_sorted[:40])
        X_all_high_score = torch.tensor(score_all_sorted[-40:])
        X_nomal_low_score = torch.tensor(score_nomal_sorted[:40])
        X_nomal_high_score = torch.tensor(score_nomal_sorted[-40:])
        X_outlier_high_score = torch.tensor(score_outlier_sorted[-40:])
        #X_gray_high_score = torch.tensor(score_gray_sorted[-50:]) #

        X_all_nomal_scores = scores[labels == 0]
        X_all_outlier_scores = scores[labels == 1]
        #X_all_gray_scores = scores[labels == 2] #

    # plot
    plot_images_grid(X_all_low, X_all_low_score, label_all_sorted[:40], 0, export_img=save_path + 'all_low_score.png', padding=2)
    plot_images_grid(X_all_high, X_all_high_score, label_all_sorted[-40:], 1, export_img=save_path + 'all_high_score.png', padding=2)
    plot_images_grid(X_normal_low, X_nomal_low_score, label_normal_sorted[:40], 0, export_img=save_path + 'nomal_low_score.png', padding=2)
    plot_images_grid(X_normal_high, X_nomal_high_score, label_normal_sorted[-40:], 1, export_img=save_path + 'nomal_high_score.png', padding=2)
    plot_images_grid(X_outlier_high, X_outlier_high_score, label_outlier_sorted[-40:], 1, export_img=save_path + 'outlier_high_score.png', padding=2)
    #plot_images_grid(X_gray_high, X_gray_high_score, label_gray_sorted[-50:], 1, export_img=save_path + 'gray_high_score', padding=2)
    plot_hist(X_all_nomal_scores, r, export_img=save_path+'normal_histogram.png', title='normal')
    plot_hist(X_all_outlier_scores, r, export_img=save_path+'outlier_histogram.png', title='outlier')
    plot_all_hist(X_all_nomal_scores, X_all_outlier_scores, r, export_img=save_path)
    #plot_all_hist(X_all_nomal_scores, X_all_outlier_scores, X_all_gray_scores, r, export_img=save_path)#
    total_time = time.time() - start_time
    logger.info('Total Time: {:.3f}s'.format(total_time))
    logger.info('Finished!')

if __name__ == '__main__':
	main()