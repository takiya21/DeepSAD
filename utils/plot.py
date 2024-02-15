# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from matplotlib import gridspec
#from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
#from matplotlib.animation import PillowWriter
#import matplotlib.patches as mpatches
#import sympy
import time
#from PIL import Image
#from numpy import random
import torch
import torchvision
#from torchvision.utils import make_grid
from sklearn.manifold import TSNE
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


class AverageMeter(object):
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

class History(object): #loss記録

    def __init__(self, keys, output_dir):
        #self.output_dir = output_dir
        self.keys = keys

        self.logs = {key: [] for key in keys}

    def __call__(self, data):
        for key, value in data.items():
            self.logs[key].append(value)

    def plot_graph(self, export_img, filename='training_curve.png'):
        start_time = time.time()
        # plt.rcParams['font.family']='Times New Roman'
        # plt.rcParams["mathtext.fontset"]="stix" 
        # plt.rcParams["font.size"]=20
        # plt.rcParams['axes.linewidth']=1.0 
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training curve.')
        loss_min = 1

        for i, (key, value) in enumerate(self.logs.items()):
            x = np.arange(len(value))
            y = np.array(value)
            ax.plot(x, y, label=key, color=cm.cividis(i / len(self.logs.keys())))
            if(i==0):
                if(loss_min > min(y)):
                    loss_min = min(y)
        ax.legend(loc='best')
        plt.ylim(0, 2)
        plt.show()
        plt.savefig(export_img + filename , transparent=False)
        plt.clf()
        plt.cla()
        plt.close('all')
        total_time = time.time() - start_time
        print("plot_graph: ",total_time)

def plot_img(dataset, export_img, filename='img.png'):
    ''' images of datasets '''
    start_time = time.time()
    t_data = np.array([td[0].to('cpu').detach().numpy().copy() for td in dataset])
    t_data = torch.tensor(np.transpose(t_data.data, (0,1,2,3)))
    if (filename!='train_img.png') & (filename!='val_img.png') & (filename!='test_img.png'):
        #mask = (dataset.label == 1)
        mask = [(label==1) for label in dataset.label]
        #data = dataset.data[mask]
        data = t_data[mask]
        #data = [dataset.data[i] for i in range(len(mask)) if mask[i]]
    else:
        data = t_data[:60]
    #print(data.shape)
    if len(data!=0):
        grid = torchvision.utils.make_grid(data, nrow=15, padding=1, normalize=True)
        #print(grid.shape)
        # plt.rcParams["figure.figsize"] = (64, 64)  #plt.rcParams["調整したいパラメータ"] = (値)でデフォルト設定をオーバーライド
        # plt.rcParams['font.family']='Times New Roman'
        # plt.rcParams["mathtext.fontset"]="stix" 
        # plt.rcParams["font.size"]=20
        # plt.rcParams['axes.linewidth']=1.0 
        plt.imshow(np.transpose(grid, (1,2,0))) #tansposeで指定した順番通りに軸を入れ替える(H, W, C)
        plt.axis('off')
        plt.savefig(export_img + filename, transparent=False)
        plt.clf()
        plt.cla()
        plt.close('all')
        total_time = time.time() - start_time
        print("plot_img: ",total_time)

def plot_images_grid(x: torch.tensor, y: torch.tensor, idx: torch.tensor, z, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    ''' sorted images by scores on testdatasets '''
    start_time = time.time()
    # plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams["mathtext.fontset"]="stix" 
    # plt.rcParams["font.size"]=20
    # plt.rcParams['axes.linewidth']=1.0 
    _, axes = plt.subplots(nrows=4, ncols=10, figsize=(15,15), sharex=False, tight_layout=True)

    for i in range(len(x)):
        if (z == 0):
            if (i<10):
                axes[0,i].imshow(x[i])
                axes[0,i].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[3,9-i].axis("off")
            elif (9<i<20):
                axes[1,i-10].imshow(x[i])
                axes[1,i-10].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[2,9-(i-10)].axis("off")
            elif (19<i<30):
                axes[2,i-20].imshow(x[i])
                axes[2,i-20].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[1,9-(i-20)].axis("off")
            elif (29<i<40):
                axes[3,i-30].imshow(x[i])
                axes[3,i-30].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[0,9-(i-30)].axis("off")
        
        else:
            if (i<10):
                axes[3,9-i].imshow(x[i])
                axes[3,9-i].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[3,9-i].axis("off")
            elif (9<i<20):
                axes[2,9-(i-10)].imshow(x[i])
                axes[2,9-(i-10)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[2,9-(i-10)].axis("off")
            elif (19<i<30):
                axes[1,9-(i-20)].imshow(x[i])
                axes[1,9-(i-20)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[1,9-(i-20)].axis("off")
            elif (29<i<40):
                axes[0,9-(i-30)].imshow(x[i])
                axes[0,9-(i-30)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[0,9-(i-30)].axis("off")
    
    plt.tight_layout()
    plt.show()

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    total_time = time.time() - start_time
    print("plot_grid_img: ",total_time)

def plot_all_hist(normal_scores, outlier_scores, r, export_img, title: str = ''):
#def plot_all_hist(normal_scores, outlier_scores, gray_scores, r, export_img, title: str = ''):

    #print(normal_scores.tolist())
    all_scores = np.concatenate((normal_scores, outlier_scores))
    all_scores = list(map(float, all_scores))
    #print(list(map(float, normal_scores)))
    scale_scores = preprocessing.minmax_scale(all_scores)
    normal_scores = scale_scores[:len(normal_scores)]
    outlier_scores = scale_scores[len(outlier_scores):]

    ''' histogram of anomaly scores '''
    import csv
    outlier_scores1 = []
    normal_scores1 = []
    for score in outlier_scores: 
        outlier_scores1.append(score)
    for score in normal_scores: 
        normal_scores1.append(score)
    with open('normal.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(normal_scores1)
    with open('outlier.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(outlier_scores1)
    f.close()
    start_time = time.time()
    # plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams["mathtext.fontset"]="stix" 
    # plt.rcParams["font.size"]=20
    # plt.rcParams['axes.linewidth']=1.0 
    fig = plt.figure(figsize=(6.4, 4.8))
    label = ['normal', 'outlier']
    #label = ['normal', 'outlier', 'gray'] #
    ax = fig.add_subplot(1,1,1) # (行, 列, 場所)
    binnum = 100
    n, bins, patches = ax.hist(outlier_scores, range=None, bins=20, alpha=0.3, color='r', label=label[1])
    n, _, _ = ax.hist(normal_scores, range=None, bins=20, alpha=0.3, color='b', label=label[0])
    #n, _, _ = ax.hist(gray_scores, range=None, bins=binnum, alpha=0.3, color='g', label=label[2]) #
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Num')
    plt.title('Histogram of Deep SAD')
    [xmin, xmax, ymin, ymax] = ax.axis()
    # ax.vlines(r, ymin=0, ymax=ymax, colors='r', linestyles='dashed', linewidth=0.5)
    #ax.set_xscale('log')
    ax.set_yticks(np.arange(0, max(n)+1, 2))
    ax.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    plt.xlim(0, 1)
    plt.show()
    fig.savefig(export_img+'histogram.png', bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    total_time = time.time() - start_time
    print("plot_all_hist: ",total_time)

def plot_hist(scores, r, export_img, title: str = ''):
    ''' outlier histogram of anomaly scores '''
    start_time = time.time()
    # plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams["mathtext.fontset"]="stix" 
    # plt.rcParams["font.size"]=20
    # plt.rcParams['axes.linewidth']=1.0 
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1,1,1) # (行, 列, 場所)
    binnum = 100
    label = title
    if (label == 'normal'):
        color = 'b'
    else:
        color = 'r'
    n, _, _ = ax.hist(scores, range=None, bins=binnum, alpha=0.3, color=color, label=label)
    plt.xlabel('anomaly scores')
    plt.ylabel('num')
    plt.title('histogram of anomaly scores')
    [_, _, _, ymax] = ax.axis()
    # ax.vlines(r, ymin=0, ymax=ymax, colors='r', linestyles='dashed', linewidth=0.5)
    #ax.set_xscale('log')
    ax.set_yticks(np.arange(0, max(n)+1, 2))
    ax.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, title='r^2: %g' % r)
    plt.show()
    fig.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    total_time = time.time() - start_time
    print("plot_hist: ",total_time)

'''
def plot_tsne(normal_scores, outlier_scores, r, export_img, title: str = ''): # 未使用
    #normal histogram of anomaly scores 
    fig = plt.figure(figsize=(6.4, 4.8))
    label = 'nomal'
    ax = fig.add_subplot(2,1,1) # (行, 列, 場所)
    normal_max = max(normal_scores)
    normal_min = min(normal_scores)
    dist = normal_max - normal_min
    binnum = int(len(normal_scores)*0.5)
    if (dist > 100):
        normal_max = 30
    if (binnum < 501):
        binnum = 500
    binnum = 100
    ax.hist(normal_scores, range=(0,int(normal_max)), bins=binnum, alpha=0.3, density=True, color='b', label=label)
    ax.set_xlabel('anomaly scores')
    ax.set_ylabel('probability')
    ax.set_title('histogram')
    [xmin, xmax, ymin, ymax] = ax.axis()
    ax.vlines(r, ymin=0, ymax=ymax, colors='r', linestyles='dashed', linewidth=0.5)
    ax.axis([-0.01, r*2.2, 0, ymax])
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, title='r: %g' % r)
    plt.show()
    if not (title == ''):
        plt.title('normal'+title)
    fig.savefig(export_img+'normal_histogram', bbox_inches='tight', pad_inches=0.1)
    plt.clf()

    # outlier histogram of anomaly scores
    fig = plt.figure(figsize=(6.4, 4.8))
    label = 'outlier'
    spec = gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[2, 1],wspace=0.1)
    ax = fig.add_subplot(spec[0]) # (行, 列, 場所)
    outlier_max = max(outlier_scores)
    outlier_min = min(outlier_scores)
    dist = outlier_max - outlier_min
    #binnum = int(len(normal_scores)*0.5)
    binnum = 100
    n, bins, patches = ax.hist(outlier_scores, range=(0,int(normal_max)), bins=binnum, alpha=0.3, density=True, color='r', label=label)
    n = list(n)
    bins = list(bins)
    plt.xlabel('anomaly scores')
    plt.ylabel('probability')
    plt.title('histogram')
    [xmin, xmax, ymin, ymax] = ax.axis()
    ax.vlines(r, ymin=0, ymax=ymax, colors='r', linestyles='dashed', linewidth=0.5)
    idx_n_max = n.index(max(n))
    bins_max = bins[idx_n_max]
    outlier_med = bins_max - bins[idx_n_max-1]
    
    if(outlier_max > outlier_med*4):
        ax2 = fig.add_subplot(spec[1], sharey=ax)
        ax2.hist(outlier_scores, range=(0,int(normal_max)), bins=binnum, alpha=0.3, density=True, color='r', label=label)
        ax2.vlines(r, ymin=0, ymax=ymax, colors='r', linestyles='dashed', linewidth=0.5)
        # left subplot
        l = abs(bins_max - outlier_min)*3
        if (outlier_min+l < r):
            ax.axis([outlier_min,r+l,0,ymax])
        else:
            ax.axis([outlier_min,outlier_min+l,0,ymax])
        # right subplot
        l = xmax/2
        ax2.axis([xmax-l,xmax,0,ymax])
        # 左のプロット領域右辺を非表示
        ax.spines['right'].set_visible(False)
        # 右のプロット領域左辺を非表示、Y軸の目盛とラベルを非表示
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(axis='y', which='both', left=False, labelleft=False) 

        d1 = 0.05 # Y軸のはみだし量
        d2 = 0.03 # 波線の高さ
        wn = 41   # 波線の数（奇数値を指定）

        pp = (0,d2,0,-d2)
        pp2 = (0,0,0,0)
        py = np.linspace(-d1,1+d1,wn)
        px = np.array([1+pp[i%4] for i in range(0,wn)])
        py2 = np.linspace(-0.05,1+0.05,wn)
        px2 = np.array([1+pp2[i%4] for i in range(0,wn)])
        p = Path(list(zip(px,py)), [Path.MOVETO]+[Path.CURVE3]*(wn-1))
        p2 = Path(list(zip(px2,py2)), [Path.MOVETO]+[Path.CURVE3]*(wn-1))

        line1 = mpatches.PathPatch(p2,lw=30, edgecolor='white',
                                facecolor='None', clip_on=False,
                                transform=ax.transAxes, zorder=10,
                                capstyle='round')

        line2 = mpatches.PathPatch(p, lw=5, edgecolor='black',
                                facecolor='None', clip_on=False,
                                transform=ax.transAxes, zorder=10)

        line3 = mpatches.PathPatch(p,lw=4, edgecolor='white',
                                facecolor='None', clip_on=False,
                                transform=ax.transAxes, zorder=10,
                                capstyle='round')

        ax.add_patch(line1)
        ax.add_patch(line2)
        ax.add_patch(line3)
        ax2.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes, title='r: %g' % r)
    
    else:
        plt.rcParams["figure.figsize"] = (6.4*2, 4.8*2)
        ax.axis([outlier_min, xmax, 0, ymax])
        ax.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, title='r: %g' % r)
     
    plt.show()
    if not (title == ''):
        plt.title('outlier'+title)
    fig.savefig(export_img+'outlier_histogram', bbox_inches='tight', pad_inches=0.1)
    plt.clf()
'''

def plot_auc(x, y, z, export_img, xlabel: str = '', ylabel: str = '', title: str = ''):
    ''' save auc curve '''
    start_time = time.time()
    if (title == 'roc_curve'):
        zlabel = 'fpr@tpr=1.0'
    else:
        zlabel = 'precision@recall=1.0'
    # plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams["mathtext.fontset"]="stix" 
    # plt.rcParams["font.size"]=20
    # plt.rcParams['axes.linewidth']=1.0 
    plt.figure(dpi=100, figsize=(10,5))
    plt.plot(x, y)
    plt.xlim(-0.03,1.03)
    plt.ylim(-0.03,1.03)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    # plt.legend(loc='best', title='%s: %.4f' % (zlabel, z), labels='')
    plt.title(title,fontname="Times New Roman")
    plt.show()
    plt.savefig(export_img + title + '.png', bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    total_time = time.time() - start_time
    print("plot_auc: ",total_time)

def plot_tsne(x, labels, scores, n_components, export_img, title: str = ''):
    start_time = time.time()
    #print(x[0])
    #x = np.array([[0.01, 0.011, -0.008, 0.13],[-0.04, 0.1, -0.02, 0.009]])
    #print(x)
    tsne = TSNE(n_components=n_components, random_state=1, perplexity = 30, n_iter = 1000)
    x = x.to('cpu')
    x_reduced = tsne.fit_transform(x)
    #print(x_reduced) #できてない、あきらめ
    
    
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(0,0,0, c='r', alpha=0.5)
    
    #print(x[:, 0])

    def plot_graph():
        for i in range(2):
            mask = (labels == i)
            target = x_reduced[mask]
            #target = x[mask]

            ax.scatter(target[:, 0], target[:, 1], target[:, 2], label=str(i), alpha=0.5)

    """ 対数尤度関数を偏微分してパラメータ推定 """
    '''
    # 変数を定義（v=σ**2としておく）
    (mu, v, y) = sympy.symbols('mu v y')
    # 尤度p(パラメータ|x)を定義
    fe=(1/sympy.sqrt(2*sympy.pi*v))*sympy.exp(-(y-mu)**2/(2*v))
    # 対数化
    logf=sympy.log(fe)
    # fを偏微分して、式を展開
    pdff1 = sympy.expand(sympy.diff(logf, mu)) #μについて偏微分
    pdff2 = sympy.expand(sympy.diff(logf, v)) #vについて偏微分

    def L_sympy(fmu,fs,var,values):       
        likelihood_mu = 0 #尤度の初期値
        likelihood_s = 0 #尤度の初期値
        for i in np.arange(len(values)):
            # likelihood
            likelihood_mu += fmu.subs(var,values[i]) #xに値を代入
            likelihood_s += fs.subs(var,values[i]) #xに値を代入
        param = sympy.solve([likelihood_mu,likelihood_s]) #方程式を解く
        return param

    mask = (labels==0)
    normal_scores = scores[mask]
    parameters = L_sympy(pdff1,pdff2,"y",normal_scores)
    parameters[0]["s"]=sympy.sqrt(parameters[0][v]) # mu: 平均 v: 分散 s: 標準偏差
    #print(parameters)      # [{}]
    print(parameters[0])    # {}
    print(parameters[0][mu])

    # ノードにscoreを表示する
    normal_scores = normal_scores.tolist()
    for n in normal_scores:
        ax.text(n)
    '''
    
    # 引数を受け取って図を回転させる関数を準備
    def plt_graph3d(angle):
        ax.view_init(30, angle)

    # アニメーションを作成
    ani = FuncAnimation(
        fig,
        func=plt_graph3d,
        frames=100,
        init_func=plot_graph,
        interval=120
    )

    plt.title(title)
    plt.show()
    export_img = "%s" % export_img+"t-SNE.gif"
    ani.save("%s" % export_img, writer="Pillow")
    plt.clf()
    total_time = time.time() - start_time
    print("plot_tsne: ",total_time)

def plot_rec(x, y, idx, z, export_img, title: str = ''):
    ''' reconstraction images '''
    start_time = time.time()
    # plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams["mathtext.fontset"]="stix" 
    # plt.rcParams["font.size"]=20
    # plt.rcParams['axes.linewidth']=1.0 
    _, axes = plt.subplots(nrows=4, ncols=10, figsize=(15,15), sharex=False, tight_layout=True)

    for i in range(len(x)):
        if (z == 0):
            if (i<10):
                axes[0,i].imshow(x[i])
                axes[0,i].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[3,9-i].axis("off")
            elif (9<i<20):
                axes[1,i-10].imshow(x[i])
                axes[1,i-10].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[2,9-(i-10)].axis("off")
            elif (19<i<30):
                axes[2,i-20].imshow(x[i])
                axes[2,i-20].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[1,9-(i-20)].axis("off")
            elif (29<i<40):
                axes[3,i-30].imshow(x[i])
                axes[3,i-30].set_title(str(i)+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[0,9-(i-30)].axis("off")
        
        else:
            if (i<10):
                axes[3,9-i].imshow(x[i])
                axes[3,9-i].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[3,9-i].axis("off")
            elif (9<i<20):
                axes[2,9-(i-10)].imshow(x[i])
                axes[2,9-(i-10)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[2,9-(i-10)].axis("off")
            elif (19<i<30):
                axes[1,9-(i-20)].imshow(x[i])
                axes[1,9-(i-20)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[1,9-(i-20)].axis("off")
            elif (29<i<40):
                axes[0,9-(i-30)].imshow(x[i])
                axes[0,9-(i-30)].set_title(str(abs(i-39))+': '+str(round(y[i].item(), 4))+' ('+str(idx[i].item())+')')
                axes[0,9-(i-30)].axis("off")
    
    plt.tight_layout()
    plt.show()

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    total_time = time.time() - start_time
    print("plot_rec: ",total_time)