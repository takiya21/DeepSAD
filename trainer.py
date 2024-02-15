# coding:utf-8
import numpy as np
import torch
import logging
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from networks.channel3_net import channel3_LeNet_Encoder
from networks.preprocess_Net import PREPROCESS_Net
from utils.plot import AverageMeter, History, plot_img, plot_auc, plot_tsne

class Trainer():
    def __init__(self, dataset_name, save_path, device, dim, rep_dim, height, width, n_epochs, lr, lr_milestone, batch_size,
                 weight_decay, normal_class, known_outlier_class):
        super().__init__()

        self.dataset_name = dataset_name
        self.save_path = save_path
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
        self.normal_class = normal_class
        self.known_outlier_class = known_outlier_class
    # }} __init__

    def train(self, train_dataset, val_dataset, classes, pretrain, nu, eta):
        logger = logging.getLogger()

        eps = 0.1
        j = 0
        for train_dataset, val_dataset in zip(train_dataset, val_dataset):
            if (j==2): # main.py の t(交差検証法のナンバー)
                # set net
                net = PREPROCESS_Net(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width).to(self.device)
                if (pretrain):
                    # initialing network parameter from ae_net's encoder　# 事前学習済みAEの重みを利用
                    net_dict = net.state_dict()
                    # Filter out decoder network keys
                    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict} # AEのエンコーダのみ取り出す
                    # Overwrite values in the existing state_dict
                    net_dict.update(ae_net_dict)
                    # Load the new state_dict
                    net.load_state_dict(net_dict)
                # set optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                # set learning rate scheduler
                #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1) # milestonesはlr変更のエポック数またはリスト
                #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

                # load datasets
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)
                # initialize center c
                c = Trainer.init_center_c(train_loader, net, eps=0.1, device=self.device)
              
                # save datasets images
                #plot_img(train_dataset, self.save_path, filename='train_img.png')
                #plot_img(val_dataset, self.save_path, filename='val_img.png')
                plot_img(train_dataset, self.save_path, filename='train_outlier_img_%s.png' % j) # trainの異常画像の保存
                
                # Training. {{{
                # =====
                # memory losses
                train_history = History(keys=('train_loss', '_t'), output_dir=None)
                val_history = History(keys=('val_loss', '_t'), output_dir=None)
                logger.info('================= Training %s =================' % j)
                logger.info('Training Start')
                best_acc = 0
                for epoch in range(self.n_epochs):
                    loop = tqdm(train_loader, unit='batch', desc='Epoch {:>3}'.format(epoch+1))
                    correct = 0
                    total = 0
                    train_idx_label_score = []
                    val_idx_label_score = []

                    # Train Step. {{{
                    # =====
                    net.train()
                    for _, batch in enumerate(loop):
                        G_meter = AverageMeter()
                        inputs, labels, idx = batch  # train_loaderの最初のミニバッチ
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        idx = idx.to(self.device)

                        # Update network. {{{
                        # =====
                        # forward network
                        outputs = net(inputs)
                        dist = torch.sum((outputs - c) ** 2, dim=1).to(self.device) # 中心からの距離の2乗
                        losses = torch.where(labels == 0, dist, eta * ((dist + eps) ** (-1))).to(self.device) # 正常ならdist, 異常ならdistの逆数(マージン付き)
                        loss = torch.mean(losses).to(self.device)
                        scores = dist

                        train_idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    scores.cpu().data.numpy().tolist()))
                        
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

                    # Calcurate Accuracy. {{{
                    _, labels, scores = zip(*train_idx_label_score)
                    labels = np.array(labels)
                    scores = np.array(scores)
                    mask = (labels == self.normal_class)
                    get_r_scores = scores[mask] # 正常ラベルのデータのみ抽出(半径取得用)
                    get_r_scores = np.array(get_r_scores)
                    r = Trainer.get_radius(get_r_scores, nu) # 半径取得

                    #train_auc = roc_auc_score(labels, scores)  # ラベル１くらすでは無理
                    for i in range(len(labels)):
                        if (scores[i] <= r and labels[i] == 0):
                            correct += 1
                        elif(scores[i] > r and labels[i] == 1):
                            correct += 1
                    total = len(labels)
                    # }}}

                    # Print training log. {{{
                    # =====
                    msg = "[Train {}] Epoch {}/{}".format('CNN', epoch + 1, self.n_epochs)
                    msg += " - {}: {:.3f}".format('train_loss', G_meter.avg)
                    msg += " - {}: {:.3f}".format('train_accuracy', correct / total)
                    msg += f" - learning rate : {scheduler.get_last_lr()[0]:.6f}"
                    logger.info(msg)
                    # }}}

                    # Validation Step. {{{
                    # =====
                    class_correct = list(0. for i in range(len(classes)))
                    class_total = list(0. for i in range(len(classes)))
                    with torch.no_grad(): #勾配の処理をしなくなる
                        net.eval()
                        loop_val = tqdm(val_loader, unit='batch', desc='Epoch {:>3}'.format(epoch + 1))
                        correct = 0
                        total = 0
                        for _, batch in enumerate(loop_val):
                            G_meter = AverageMeter()
                            inputs, labels, idx = batch
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            idx = idx.to(self.device)

                            outputs = net(inputs)
                            scores = torch.sum((outputs - c) ** 2, dim=1).to(self.device)
                            loss = torch.mean(scores).to(self.device)

                            val_idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    scores.cpu().data.numpy().tolist()))

                            G_meter.update(loss.item(), inputs[0].size()[0])
                            val_history({'val_loss': loss.item()})
                    # }}}

                    _, labels, scores = zip(*val_idx_label_score)

                    labels = np.array(labels)
                    scores = np.array(scores)

                    logger.info('r: %.2f' % r)

                    for i in range(len(labels)):
                        if(labels[i] == 0):
                            class_total[0] += 1
                            if (scores[i] < r): # correct nomal
                                class_correct[0] += 1
                        
                        elif(labels[i] == 1):
                            class_total[1] += 1
                            if(scores[i] >= r): # correct outlier
                                class_correct[1] += 1

                    # Print validation log. {{{
                    # =====
                    msg = "[Validation {}] Epoch {}/{}".format(
                        'CNN', epoch + 1, self.n_epochs)
                    msg += " - {}: {:.3f}".format('val_loss', G_meter.avg)
                    val_acc = sum(class_correct) / sum(class_total)
                    msg += " - {}: {:.3f}".format('val_accuracy', val_acc)
                    logger.info(msg)
                    if best_acc < val_acc:
                        torch.save(net.state_dict(), self.save_path+'best_model_%s.pth' % j)
                        val_acc = best_acc
                    # }}}
                train_history.plot_graph(self.save_path, filename='train_curve_%s.png' % j)
                val_history.plot_graph(self.save_path, filename='val_curve_%s.png' % j)
                # netのネットワーク重み保存
                torch.save(net.state_dict(), self.save_path+'model_%s.pth' % j)
                # }}} epoch
            # }}} dataset
            j += 1
        return c
    # }}} train
    
    def test(self, test_dataset, classes, c):
        logger = logging.getLogger()
        # load test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4, drop_last=False)
        # save test images
        #plot_img(test_dataset, self.save_path, filename='test_img.png')
        # set net
        net= PREPROCESS_Net(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width).to(self.device)
        # initialing network parameter
        #net_dict = net.state_dict()
        # Overwrite values in the existing state_dict
        #net_dict.update(train_net_dict)
        # Load the new state_dict
        #net.load_state_dict(train_net_dict)
        j = 2
        net.load_state_dict(torch.load(self.save_path+'model_%s.pth' % j))
        # Test
        outputs = 0
        class_correct = list(0. for _ in range(len(classes)))
        class_total = list(0. for _ in range(len(classes)))
        test_idx_label_score = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = net(inputs) #ネットに通す

                scores = torch.sum((outputs - c) ** 2, dim=1).to(self.device)

                test_idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

            idx, labels, scores = zip(*test_idx_label_score)
            idx, labels, scores = np.array(idx), np.array(labels), np.array(scores)

            #'''
            test_auc_roc = roc_auc_score(labels, scores) # roc_auc_score(正解ラベル, 予測スコア)
                                                            # 予測スコアは確率でなくてもよい(スケールは0~1でなくてもよい)
                                                            # 閾値は自動で計算される(閾値を指定する方法もあり)
        
            fpr, tpr, thresholds = roc_curve(labels, scores)
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            test_auc_pr = metrics.auc(recall, precision)

            tpr_1 = np.where(tpr==1.0)
            recall_1 = np.where(recall==1.0)
            fpr_1 = fpr[tpr_1[0][0]]
            precision_1 = precision[recall_1[-1][0]]
            #'''

            mask = (labels == self.known_outlier_class)
            get_r_scores = scores[mask] # 異常ラベルのスコアのみ抽出(半径取得用)
            r = min(get_r_scores) # 異常データの異常スコアの最小値を半径として評価(異常の見逃し0)
            logger.info('r^2: %.4f' % r)
            
            mask = (labels == self.normal_class)
            get_m_scores = scores[mask] # 正常ラベルのスコアのみ抽出
            m = max(get_m_scores) # 正常データの異常スコアの最大値を取得(マージン計算用)

            for i in range(len(labels)):
                if(labels[i] == 0):
                    class_total[0] += 1
                    if (scores[i] < r): # correct nomal
                        class_correct[0] += 1
                
                elif(labels[i] == 1):
                    class_total[1] += 1
                    if(scores[i] >= r): # correct outlier
                        class_correct[1] += 1

            # Log results
            logger.info('Test result')
            logger.info('Test AUC_ROC: %.2f' % (100 * test_auc_roc))#
            logger.info('Test AUC_PR: %.2f' % (100 * test_auc_pr))#
            logger.info('fpr@tpr=1.00: %.4f' % fpr_1)#
            logger.info('margin: %.4f' % (r-m))
            
            #'''
            plot_auc(fpr, tpr, fpr_1, self.save_path, xlabel= 'FPR: False positive rate',
                        ylabel= 'TPR: True positive rate', title= 'roc_curve')
            plot_auc(recall, precision, precision_1, self.save_path, xlabel= 'recall',
                        ylabel= 'precision', title= 'pr_curve')
            #'''
            #plot_tsne(outputs, labels, scores, 3, save_path, title= 't-SNE')
        
        logger.info('total_correct: %d' % sum(class_correct))
        logger.info('total: %d' % sum(class_total))
        logger.info('normal_correct: %d' % class_correct[0])
        logger.info('normal_total: %d' % class_total[0])
        logger.info('outlier_correct: %d' % class_correct[1])
        logger.info('outlier_total: %d' % class_total[1])
        logger.info('Accuracy of the network on the test images: %d %%' % (100 * sum(class_correct) / sum(class_total)))
        for i in range(len(classes)):
            logger.info('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
        return idx, labels, scores, r, c
    # }}} test

    def init_center_c(loader, net, eps, device): # 半径初期化(フォーワードパスの平均)
        ''' initialing center c '''
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=device) # (0, 0, ..., 0) rep_dim個
        net.eval()
        with torch.no_grad():
            for data in loader:
                inputs, _, _ = data
                inputs = inputs.to(device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
            
        c /= n_samples
        #print("#############")
        #print(c)
        # If c_i is too close to 0, set to +-eps.  #超球崩壊を防ぐため中心を0から遠ざける処理
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    # }} init_center_c

    def get_radius(get_r_scores, nu): # get radius
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        # trainの異常スコアの平方根をソートし, nu分位目を半径
        get_r_scores = np.sqrt(get_r_scores)
        get_r_scores = sorted(get_r_scores)
        r = np.quantile(get_r_scores, nu)
        return r
    #}} get_radius
# }}} Trainer