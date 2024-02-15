# coding:utf-8
import torch #機械学習ライブラリ
import torch.nn as nn #ニューラルネットワーク
import torch.nn.functional as F

class EncoderBlock(nn.Module):
  def __init__(self, in_feature, out_feature):
    super(EncoderBlock, self).__init__()
    self.in_feature = in_feature
    self.out_feature = out_feature

    layers = []
    layers.append(nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=5, bias=False, stride=1, padding=2))
    layers.append(nn.BatchNorm2d(out_feature, eps=1e-04, affine=False))
    layers.append(nn.LeakyReLU())
    layers.append(nn.MaxPool2d(2, 2))
    self.block = nn.Sequential(*layers)

  def forward(self, x):
    return self.block(x)

'''
class DecoderBlock(nn.Module):
  def __init__(self, in_feature, out_feature):
    super(EncoderBlock, self).__init__()
    self.in_feature = in_feature
    self.out_feature = out_feature

    layers = []

  def forward(self, x):
    return self.block(x)
'''

class Encoder(nn.Module):
    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width

        self.conv1 = EncoderBlock(3, dim)
        #self.conv2 = EncoderBlock(dim, dim)
        #self.conv3 = EncoderBlock(dim, dim)
        self.conv4 = EncoderBlock(dim, dim*2)
        #self.conv5 = EncoderBlock(dim*2, dim*2)
        #self.conv6 = EncoderBlock(dim*2, dim*2)
        self.conv7 = EncoderBlock(dim*2, dim*4)
        #self.conv8 = EncoderBlock(dim*4, dim*4)
        #self.conv9 = EncoderBlock(dim*4, dim*4)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.pool(x)

        x = self.conv4(x)
        #print(x.size())
        #x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.pool(x)

        x = self.conv7(x)
        #print(x.size())
        #x = self.conv8(x)
        #x = self.conv9(x)
        #x = self.pool(x)
        #para = x.size()[2]*x.size()[3] > 82

        return x

class Decoder(nn.Module):
    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width

        #self.deconv0 = nn.ConvTranspose2d(dim*8, dim*8, 5, bias=False, padding=2)
        #nn.init.xavier_uniform_(self.deconv0.weight, gain=nn.init.calculate_gain('leaky_relu'))
        #self.bn2d2 = nn.BatchNorm2d(dim*8, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(dim*4, dim*4, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(dim*4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(dim*4, dim*2, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(dim*2, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(dim*2, dim, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(dim, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(dim, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc1 = nn.Linear(int(((dim*4 * int(height/2/2/2) * int(width/2/2/2))+self.rep_dim)/2), dim*4 * int(height/2/2/2) * int(width/2/2/2), bias=False)
        self.fc2 = nn.Linear(rep_dim, int(((dim*4 * int(height/2/2/2) * int(width/2/2/2))+rep_dim)/2), bias=False)
        #self.fc3 = nn.Linear(int(((dim*8 * int(height/2/2/2/2) * int(width/2/2/2/2))+rep_dim)/2), dim*8 * int(height/2/2/2/2) * int(width/2/2/2/2), bias=False)
        #self.fc4 = nn.Linear(rep_dim, int(((dim*8 * int(height/2/2/2/2) * int(width/2/2/2/2))+rep_dim)/2), bias=False)

    def forward(self, x):
        
        #print('endoder_last: ', x.size()) # [32, 16] (バッチサイズ, rep_dim) 32×32
        x = self.fc2(x)
        #print('fc2', x.size()) # [32, 2056]
        x = self.fc1(x)
        #print('fc1', x.size()) # [32, 4096]
        x = x.view(int(x.size(0)), self.dim*4, int(self.height/2/2/2), int(self.width/2/2/2))
        #print('1', x.size()) # [32, 256, 4, 4]
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        #print('2', x.size()) # [32, 256, 4, 4]
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        #print('3', x.size()) # [32, 256, 8, 8]
        x = self.deconv2(x)
        #print('4', x.size()) # [32, 128, 8, 8]
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        #print('5', x.size()) # [32, 128, 16, 16]
        x = self.deconv3(x)
        #print('6', x.size()) # [32, 64, 16, 16]
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        #print('7', x.size()) # [32, 64, 32, 32]
        x = self.deconv4(x) 
        #print('8', x.size()) # [32, 3, 32, 32]
        x = torch.sigmoid(x)
        #print('9', x.size()) # [32, 3, 32, 32]
        
        return x

class channel3_LeNet_Autoencoder(nn.Module):
    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width

        self.encoder = Encoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width)
        self.decoder = Decoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width)
        self.fc1 = nn.Linear(dim*4 * int(height/2**3) * int(width/2**3), int(((dim*4 * int(height/2**3) * int(width/2**3))+rep_dim)/2), bias=False)
        self.fc2 = nn.Linear(int(((dim*4 * int(height/2**3) * int(width/2**3))+rep_dim)/2), rep_dim, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.decoder(x)
        return x

class channel3_LeNet_Encoder(nn.Module):
    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width

        self.encoder = Encoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width)
        self.fc1 = nn.Linear(dim*4 * int(height/2**3) * int(width/2**3), int(((dim*4 * int(height/2**3) * int(width/2**3))+rep_dim)/2), bias=False)
        self.fc2 = nn.Linear(int(((dim*4 * int(height/2**3) * int(width/2**3))+rep_dim)/2), rep_dim, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class channel3_LeNet_Encoder2(nn.Module):
    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.dim = dim
        self.rep_dim = rep_dim
        self.height = height
        self.width = width

        self.encoder = Encoder(dim=self.dim, rep_dim=self.rep_dim, height=self.height, width=self.width)
        self.conv4 = EncoderBlock(dim*4, dim*8)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.Linear(dim*8 * int(height/2**4) * int(width/2**4), int(((dim*8 * int(height/2**4) * int(width/2**4))+rep_dim)/2), bias=False)
        self.fc4 = nn.Linear(int(((dim*8 * int(height/2**4) * int(width/2**4))+rep_dim)/2), rep_dim, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv4(x)
        #x = self.pool(x)
        x = x.view(int(x.size(0)), -1)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
