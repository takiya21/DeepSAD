# coding:utf-8
from networks.channel3_net import channel3_LeNet_Autoencoder
from utils.plot import AverageMeter, History, plot_auc, plot_images_grid
#from networks.initializer import init_weight

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
#import torch.optim as optim
import numpy as np

class AETrainer():
    def __init__(self, dataset_name, save_path, device, dim, rep_dim, height, width, n_epochs, lr, lr_milestone, batch_size, weight_decay):
        super().__init__()

        self.dataset_name = dataset_name
        self.save_path = save_path + ('/pretrain/')
        self.device = device
        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        os.makedirs(self.save_path, exist_ok=True)

    def train(self, train_dataset, val_dataset, classes):
        logger = logging.getLogger()

        j = 0
        # memory losses
        train_history = History(keys=('train_loss', '_t'), output_dir=None)
        val_history = History(keys=('val_loss', '_t'), output_dir=None)
        for train_dataset, val_dataset in zip(train_dataset, val_dataset):
            if (j==0):
                # set net
                ae_net = channel3_LeNet_Autoencoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width).to(self.device)
                # set optimizer
                optimizer = torch.optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                # set learning rate scheduler
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1) # milestonesはlr変更のエポック数またはリスト
                # Set loss
                criterion = nn.MSELoss(reduction='none')
                # Set device for network
                ae_net = ae_net.to(self.device)
                criterion = criterion.to(self.device)
                # set gauss
                #gauss = np.random.normal(0,30,(self.batch_size, 3, self.height ,self.width)) # 3 channel用

                # load datasets
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
                # Training. {{{
                # =====
                # memory losses
                train_history = History(keys=('train_loss', '_t'), output_dir=None)
                val_history = History(keys=('val_loss', '_t'), output_dir=None)
                logger.info('================= Pretraining %s =================' % j)
                logger.info('Training Start')
                for epoch in range(self.n_epochs):
                    loop = tqdm(train_loader, unit='batch', desc='Epoch {:>3}'.format(epoch+1))
                    correct = 0
                    total = 0
                    train_idx_label_score = []
                    val_idx_label_score = []

                    # Train Step. {{{
                    # =====
                    ae_net.train()
                    for _, batch in enumerate(loop):
                        G_meter = AverageMeter()
                        inputs, labels, idx = batch  # train_loaderの最初のミニバッチ
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        idx = idx.to(self.device)

                        # Update network. {{{
                        # =====
                        # forward network
                        # Update network parameters via backpropagation: forward + backward + optimize
                        #rec = ae_net(inputs, self.dim, self.height, self.width)
                        rec = ae_net(inputs)
                        rec_loss = torch.where(labels==0, torch.mean(criterion(rec, inputs), dim=tuple(range(1, rec.dim()))), torch.mean(criterion(rec, inputs), dim=tuple(range(1, rec.dim())))**(-1))
                        loss = torch.mean(rec_loss)
                        
                        # backward network
                        optimizer.zero_grad()   # 勾配初期化
                        loss.backward() # 勾配計算
                        optimizer.step() # パラメータ更新
                        scheduler.step()
                        # }}}

                        # Get losses. {{{
                        G_meter.update(loss.item(), inputs[0].size()[0])
                        train_history({'train_loss': loss.item()}) # __call__が呼ばれる
                        # }}}
                        # }}}

                    # Print training log. {{{
                    # =====
                    msg = "[Train {}] Epoch {}/{}".format('CNN', epoch + 1, self.n_epochs)
                    msg += " - {}: {:.3f}".format('train_loss', G_meter.avg)
                    logger.info(msg)
                    # }}}

                    # Validation Step. {{{
                    # =====
                    class_correct = list(0. for i in range(len(classes)))
                    class_total = list(0. for i in range(len(classes)))
                    with torch.no_grad(): #勾配の処理をしなくなる
                        ae_net.eval()
                        loop_val = tqdm(val_loader, unit='batch', desc='Epoch {:>3}'.format(epoch + 1))
                        correct = 0
                        total = 0

                        for _, batch in enumerate(loop_val):
                            G_meter = AverageMeter()
                            inputs, labels, idx = batch
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            idx = idx.to(self.device)

                            rec = ae_net(inputs)
                            rec_loss = torch.mean(criterion(rec, inputs), dim=tuple(range(1, rec.dim())))
                            loss = torch.mean(rec_loss)
                            #scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))
                            
                            '''
                            val_idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    scores.cpu().data.numpy().tolist()))
                            '''

                            G_meter.update(loss.item(), inputs[0].size()[0])
                            val_history({'val_loss': loss.item()})
                        # }}}

                        # Print validation log. {{{
                        # =====
                        msg = "[Validation {}] Epoch {}/{}".format(
                            'CNN', epoch + 1, self.n_epochs)
                        msg += " - {}: {:.3f}".format('val_loss', G_meter.avg)
                        logger.info(msg)
                        # }}}
                train_history.plot_graph(self.save_path, filename='train_curve_%s.png' % j)
                val_history.plot_graph(self.save_path, filename='val_curve_%s.png' % j)

                # ae_netのネットワーク重み保存
                torch.save(ae_net.state_dict(), self.save_path+'model_%s.pth' % j)

                j += 1
                # }}} epoch
            # }}} j
        #}}}} dataset

    def test(self, test_dataset, classes):
        logger = logging.getLogger()

        # load test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4, drop_last=False)
        # set net
        ae_net = channel3_LeNet_Autoencoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width).to(self.device)
        # Load the new state_dict
        j = 0
        ae_net.load_state_dict(torch.load(self.save_path+'model_%s.pth' % j))
        # Set loss
        criterion = nn.MSELoss(reduction='none')
        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # ガウシアンノイズ
        def addGaussianNoise(src):
            batch_size, ch, row, col= src.shape
            mean = 0
            sigma = 30
            src = src.cpu().numpy()
            gauss = np.random.normal(mean,sigma,(batch_size, ch, row,col)) # batch_size = row, col
            gauss = gauss * 10
            noisy = src + gauss # ガウスノイズ付加
            
            def minmax(x):
                if x > 255:
                    x = 255
                elif x < 0:
                    x = 0
                return x
            
            noisy = np.array(noisy, dtype=np.int64)
            noisy = map(minmax, noisy)
            noisy = torch.from_numpy(gauss).float().to(self.device)

            return noisy

        # Test
        correct = 0
        total = 0
        test_idx_label_score = []
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        outputs = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device) #gpuに通す
                labels = labels.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                test_idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            rec.cpu().data.numpy().tolist()))

            idx, labels, scores, rec = zip(*test_idx_label_score)
            idx, labels, scores, rec = np.array(idx), np.array(labels), np.array(scores), torch.from_numpy(np.array(rec))
            fpr, tpr, thresholds = roc_curve(labels, scores)
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            test_auc_roc = roc_auc_score(labels, scores)
            test_auc_pr = metrics.auc(recall, precision)

            # Log results
            logger.info('Test result')
            logger.info('Test AUC_ROC: %.2f' % (100 * test_auc_roc))
            logger.info('Test AUC_PR: %.2f' % (100 * test_auc_pr))
        
        idx_all_sorted = idx[np.argsort(scores)]  # from lowest to highest score
        idx_normal_sorted = idx[labels == 0][np.argsort(scores[labels == 0])]
        idx_outlier_sorted = idx[labels == 1][np.argsort(scores[labels == 1])]

        score_all_sorted = scores[np.argsort(scores)]
        score_nomal_sorted = scores[labels == 0][np.argsort(scores[labels == 0])]
        score_outlier_sorted = scores[labels == 1][np.argsort(scores[labels == 1])]

        label_all_sorted = labels[idx_all_sorted]
        label_normal_sorted = labels[idx_normal_sorted]
        label_outlier_sorted = labels[idx_outlier_sorted]

        if (self.dataset_name != 'mnist'):
            t_data = np.array([td[0].to('cpu').detach().numpy().copy() for td in test_dataset])
            t_data = t_data[:,0]

            X_all_low = t_data[idx_all_sorted[:40]]
            X_all_high = t_data[idx_all_sorted[-40:]]
            X_normal_low = t_data[idx_normal_sorted[:40]]
            X_normal_high = t_data[idx_normal_sorted[-40:]]
            X_outlier_high = t_data[idx_outlier_sorted[-40:]]
            rec_outlier_high = np.transpose(rec.data[idx_outlier_sorted, ...], (0,2,3,1)).clone().detach()
            rec_all_low = np.transpose(rec.data[idx_all_sorted[:40], ...], (0,2,3,1)).clone().detach()
            rec_all_high = np.transpose(rec.data[idx_all_sorted[-40:], ...], (0,2,3,1)).clone().detach()

            X_all_low_score = torch.tensor(np.transpose(score_all_sorted[:40]))
            X_all_high_score = torch.tensor(np.transpose(score_all_sorted[-40:]))
            X_nomal_low_score = torch.tensor(np.transpose(score_nomal_sorted[:40]))
            X_nomal_high_score = torch.tensor(np.transpose(score_nomal_sorted[-40:]))
            X_outlier_high_score = torch.tensor(np.transpose(score_outlier_sorted[-40:]))

        # plot
        plot_images_grid(X_all_low, X_all_low_score, label_all_sorted[:40], 0, export_img=self.save_path + 'all_low_score', padding=2)
        plot_images_grid(X_all_high, X_all_high_score, label_all_sorted[-40:], 1, export_img=self.save_path + 'all_high_score', padding=2)
        plot_images_grid(rec_all_high, X_all_high_score, label_all_sorted[-40:], 1, export_img=self.save_path + 'rec_all_high_score', padding=2)
        plot_images_grid(rec_all_low, X_all_low_score, label_all_sorted[:40], 0, export_img=self.save_path + 'rec_all_low_score', padding=2)
        plot_images_grid(X_outlier_high, X_outlier_high_score, label_outlier_sorted[-40:], 1, export_img=self.save_path + 'outlier_high_score', padding=2)
        plot_images_grid(rec_outlier_high, X_outlier_high_score, label_outlier_sorted[-40:], 1, export_img=self.save_path + 'rec_outlier_high_score', padding=2)