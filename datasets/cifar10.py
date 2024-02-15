# coding:utf-8
import os #OSに依存する機能を利用(ファイル, ディレクトリなど)
import numpy as np #Pythonの数値計算ライブラリ(ベクトル, 行列など)
import matplotlib.pyplot as plt #グラフ描画ライブラリ

import torch #機械学習ライブラリ
import torch.nn as nn #ニューラルネットワーク
import torch.nn.functional as F #活性化関数や損失関数など / Fは慣習的にReLU関数
import torchvision #pytorchのコンピュータービジョン用のパッケージ / データのロードや下処理用の関数など
import torchvision.transforms as transforms #画像オブジェクトを渡すと前処理(オーグメント/正規化等)を行ってくれる
from torch.utils import data
from torchvision import datasets as datasets

from torch.utils.data.dataset import Subset

from PIL import Image #PILは画像処理ライブラ
from tqdm import tqdm #trangeはtqdm(range(i))の最適化されたインスタンス

# dataset download
def load_cifar10(root, transform, train:bool):                                         #メソッド化
    cifar10_dataset = torchvision.datasets.CIFAR10(                              #def 関数名(引数1, 引数2, ...):
                        root=root, #Datesetを参照保存するディレクトリの指定  #    処理
                        train=train,   #Training用のdataを取得するかどうか       #    return 戻り値
                        download=True, #参照したディレクトリにDatasetがない場合ダウンロードするかどうか
                        transform=transform #定義した前処理を渡す.Dataset内のdataを参照する際にその前処理を自動で行う
                        )
    return cifar10_dataset

class Mydatasets(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data)[0][idx]
          out_label = self.label[idx]
        else:
          out_data = self.data[idx]
          out_label =  self.label[idx]

        return out_data, out_label, idx

def mycifar10(root, height, width, approach: str, normal_class: int, known_outlier_class: int):

    train_transform = torchvision.transforms.Compose([ #引数で渡されたlist型の[～,～,...]を先頭から順に実行していく
                                transforms.Resize((height,width)),
                                transforms.RandomHorizontalFlip(p=0.2),
                                transforms.RandomVerticalFlip(p=0.2),
                                transforms.RandomRotation(degrees=10),
                                torchvision.transforms.RandomGrayscale(p=0.1),
                                transforms.ToTensor(), #PIL imageのdate(Height×Width×Channel)からTensor型のdata(Channel×Height×Width)に変換, [0~1]にリスケール
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1, 1]にリスケール
                                #transforms.Normalize((0.5), (0.5)),
                                #transforms.RandomErasing(p=0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3), value=0)
                                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
                                
    ])

    test_transform = torchvision.transforms.Compose([
                                transforms.Resize((height,width)),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                #transforms.Normalize((0.5), (0.5)),
    ])

    dataset = load_cifar10(root=root,transform=train_transform, train=True) #train=Trueで訓練用の50000の方をロード, Falseでテスト用の10000の方をロード
                                                            #transform=train_transformでロードした後上の下処理をする

    train_size = int(len(dataset) * 0.8) #len()でサイズ(要素数や文字数)を取得  #40000
    val_size = len(dataset) - train_size                                    #10000
    

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) #ランダムに分割
    test_dataset = load_cifar10(root=root, transform=test_transform, train=False) #テスト用の10000の方をロード
    test_size = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False, num_workers=4, drop_last=False)
    
    #print("1111111111111111111111111111111111")
    #print(test_dataset.data[0])
    data_train, label_train = next(iter(train_loader))
    data_val, label_val = next(iter(val_loader))
    data_test, label_test = next(iter(test_loader))

    #print("2222222222222222222222222222222222222")
    #print(data_test[0])

    # train data
    if (approach == 'DSVDD'):
        mask = (label_train == normal_class) # 同ラベルのデータのみ抽出
        label_train = label_train[mask]
        data_train = data_train[mask]
    elif (approach == 'DSAD'):
        mask = (label_train == normal_class) | (label_train == known_outlier_class) # 同ラベルのデータのみ抽出
        label_train = label_train[mask]
        data_train = data_train[mask]  

    '''
    for data, label in test_loader:
        print("before_test")
        print(len(label))
        print(label)
        break
    '''
    '''
    data_train, label_train = next(iter(train_loader))
    data_val, label_val = next(iter(val_loader))
    data_test, label_test = next(iter(test_loader))
    print("label_train")
    print(label_train)
    print("len")
    print(len(label_train))
    print("a")
    print(label_train[0])
    label_train[0] = 1
    print("b")
    print(label_train[0])
    '''

    # nomal: 0, outlier: 1
    for i in range(len(label_train)):
        if(label_train[i] == normal_class):
            label_train[i] = 0 # nomal
        else:
            label_train[i] = 1 # outlier
    
    for i in range(len(label_val)):
        if(label_val[i] == normal_class):
            label_val[i] = 0 # nomal
        else:
            label_val[i] = 1 # outlier
            
    for i in range(len(label_test)):
        if(label_test[i] == normal_class):
            label_test[i] = 0 # nomal
        else:
            label_test[i] = 1 # outlier

    '''
    print("##")
    print(type(train_dataset)) # Subset
    print(type(train_dataset[0])) # tuple
    print(type(train_dataset[0][0])) # Tensor
    print(type(train_dataset[0][1])) # int
    '''
    
    train_dataset = Mydatasets(data_train, label_train)
    val_dataset   = Mydatasets(data_val, label_val)
    test_dataset  = Mydatasets(data_test, label_test)

    #print("3333333333333333333333333")
    #print(test_dataset.data[0])

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = ('0', '1')

    return train_dataset, val_dataset, test_dataset, classes
    
