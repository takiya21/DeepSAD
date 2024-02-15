# coding:utf-8
import os #OSに依存する機能を利用(ファイル, ディレクトリなど)
import numpy as np #Pythonの数値計算ライブラリ(ベクトル, 行列など)
import matplotlib.pyplot as plt #グラフ描画ライブラリ

import torch #機械学習ライブラリ
import torch.nn as nn #ニューラルネットワーク
import torch.nn.functional as F #活性化関数や損失関数など / Fは慣習的にReLU関数
import torchvision #pytorchのコンピュータービジョン用のパッケージ / データのロードや下処理用の関数など
import torchvision.transforms as transforms #画像オブジェクトを渡すと前処理(オーグメント/正規化等)を行ってくれる

from PIL import Image #PILは画像処理ライブラリ
from tqdm.notebook import trange, tqdm #trangeはtqdm(range(i))の最適化されたインスタンス

# model
class EncoderBlock(nn.Module):                  #class クラス名
  def __init__(self, in_feature, out_feature):  #  def __init__(self, 他の引数)
    super(EncoderBlock, self).__init__() #継承
    self.in_feature = in_feature
    self.out_feature = out_feature

    layers = []
    layers.append(nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=1))
    #複数の入力平面で構成される入力信号に2D畳み込みを適用
    #in_channelsは入力画像のチャンネル数, out_channelsは畳み込みによって生成されたチャンネル数
    #kernel_sizeは畳み込みカーネルのサイズ
    #stridesは各次元方向に１つ隣の要素に移動するために必要なバイト数をタプルで表示したもの
    layers.append(nn.BatchNorm2d(out_feature))
    #4D入力にバッチ正規化を適用
    layers.append(nn.ReLU(inplace=True))
    #正規化線形ユニット関数を要素ごとに適用
    #inplace=Trueで追加の出力を割り当てずに、入力を直接変更
    self.block = nn.Sequential(*layers)
    #layersの値を順番に渡す

  def forward(self, x):
    return self.block(x)

class Encoder(nn.Module):
  def __init__(self, dim):
    super(Encoder, self).__init__()

    #self.conv1 = EncoderBlock(1  , dim)
    self.conv1 = EncoderBlock(3  , dim)
    self.conv2 = EncoderBlock(dim, dim)
    self.conv3 = EncoderBlock(dim, dim)
    self.conv4 = EncoderBlock(dim, dim*2)
    self.conv5 = EncoderBlock(dim*2, dim*2)
    self.conv6 = EncoderBlock(dim*2, dim*2)
    self.conv7 = EncoderBlock(dim*2, dim*4)
    self.conv8 = EncoderBlock(dim*4, dim*4)
    self.conv9 = EncoderBlock(dim*4, dim*4)
    self.pool = nn.MaxPool2d(kernel_size=2)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.pool(out) 

    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.pool(out)
    
    out = self.conv7(out)
    out = self.conv8(out)
    out = self.conv9(out)
    out = self.pool(out)

    return out

class Classifier(nn.Module):
  def __init__(self, class_num, enc_dim, rep_dim, in_w, in_h):
    super(Classifier, self).__init__()

    self.enc_dim = enc_dim
    self.rep_dim = rep_dim
    self.in_w = in_w
    self.in_h = in_h
    self.fc_dim = enc_dim*4 * int(in_h/2/2/2) * int(in_w/2/2/2) # 次元を合わせる(プーリング３回とdim*2を２回)

    self.Encoder = Encoder(self.enc_dim)
    self.fc1 = nn.Linear(self.fc_dim, 512) #nn.Linear(各入力サンプルのサイズ, 各出力サンプルのサイズ)
    #線形変換
    #self.fc2 = nn.Linear(512, 256) #512次元から256次元に変換
    #self.fc3 = nn.Linear(256, self.rep_dim)
    self.fc2 = nn.Linear(512, 512) #512次元から256次元に変換
    self.fc3 = nn.Linear(512, self.rep_dim)
  
  def forward(self, x):
    out = self.Encoder(x)
    #print('out1', out.size())
    out = out.view(-1, self.fc_dim) #平衡化と同じ
    #print('out2', out.size())
    out = F.leaky_relu(self.fc1(out))
    #print('out3', out.size())
    out = F.leaky_relu(self.fc2(out))
    #print('out4', out.size())
    out = self.fc3(out)
    #print('out5', out.size())
    return out

class CIFAR10_LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x