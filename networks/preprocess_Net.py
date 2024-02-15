import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class PREPROCESS_Net(nn.Module):

    def __init__(self, dim, rep_dim, height, width):
        super().__init__()

        self.nc = 3
        self.nf = 8
        self.rep_dim = rep_dim

        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, rep_dim)

        # 入力： (NC) x 256 x 256
        self.conv1 = nn.Conv2d(self.nc, self.nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf) x 128 x 128

        self.conv2 = nn.Conv2d(self.nf, self.nf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nf * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)            
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*2) x 64 x 64

        self.conv3 = nn.Conv2d(self.nf * 2, self.nf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nf * 4)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        # サイズ：(self.nf*4) x 32 x 32

        self.conv4 = nn.Conv2d(self.nf * 4, self.nf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nf * 8)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*8) x 16 x 16

        self.conv5 = nn.Conv2d(self.nf * 8, self.nf * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.nf * 16)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        self.pool5 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*16) x 8 x 8

        self.conv6 = nn.Conv2d(self.nf * 16, self.nf * 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.nf * 32)
        self.act6 = nn.LeakyReLU(0.2, inplace=True)
        self.pool6 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*32) x 4 x 4
        
        self.fc1 = nn.Linear(256 * 4 * 4, self.rep_dim, bias=False)



    def forward(self, x):

        x = self.model(x)

        # conv1 = self.conv1(x)
        # act1 = self.act1(conv1)
        # pool1 = self.pool1(act1)
        
        # conv2 = self.conv2(pool1)
        # bn2 = self.bn2(conv2)
        # act2 = self.act2(bn2)
        # pool2 = self.pool2(act2)
        
        # conv3 = self.conv3(pool2)
        # bn3 = self.bn3(conv3)
        # act3 = self.act3(bn3)
        # pool3 = self.pool3(act3)
        
        # conv4 = self.conv4(pool3)
        # bn4 = self.bn4(conv4)
        # act4 = self.act4(bn4)
        # pool4 = self.pool4(act4)
        
        # conv5 = self.conv5(pool4)
        # bn5 = self.bn5(conv5)
        # act5 = self.act5(bn5)
        # pool5 = self.pool5(act5)
        
        # conv6 = self.conv6(pool5)
        # bn6 = self.bn6(conv6)
        # act6 = self.act6(bn6)
        # pool6 = self.pool6(act6)
        
        # pool6 = pool6.view(pool6.size(0), -1)
        # fc1 = self.fc1(pool6)

        # x = fc1
        return x

"""
class PREPROCESS_Net_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.nc = 3
        self.nf = 8
        self.rep_dim = 256

        # 入力： (NC) x 256 x 256
        self.conv1 = nn.Conv2d(self.nc, self.nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf) x 128 x 128

        self.conv2 = nn.Conv2d(self.nf, self.nf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nf * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)            
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*2) x 64 x 64

        self.conv3 = nn.Conv2d(self.nf * 2, self.nf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nf * 4)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        # サイズ：(self.nf*4) x 32 x 32

        self.conv4 = nn.Conv2d(self.nf * 4, self.nf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nf * 8)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*8) x 16 x 16

        self.conv5 = nn.Conv2d(self.nf * 8, self.nf * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.nf * 16)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        self.pool5 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*16) x 8 x 8

        self.conv6 = nn.Conv2d(self.nf * 16, self.nf * 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.nf * 32)
        self.act6 = nn.LeakyReLU(0.2, inplace=True)
        self.pool6 = nn.AvgPool2d(kernel_size=2)
        # サイズ： (self.nf*32) x 4 x 4
        
        self.fc1 = nn.Linear(256 * 4 * 4, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 256 * 4 * 4, bias=False)
        # サイズ： (self.nf*32) x 4 x 4


        self.convt7 = nn.ConvTranspose2d( self.nf * 32, self.nf * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(self.nf * 16)
        self.act7 = nn.LeakyReLU(0.2, inplace=True)
        # サイズ： (ndf*16) x 8 x 8

        self.convt8 = nn.ConvTranspose2d( self.nf * 16, self.nf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(self.nf * 8)
        self.act8 = nn.LeakyReLU(0.2, inplace=True)
        # サイズ：(self.nf*8) x 16 x 16

        self.convt9 = nn.ConvTranspose2d(self.nf * 8, self.nf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(self.nf * 4)
        self.act9 = nn.LeakyReLU(0.2, inplace=True)
        # サイズ：(self.nf*4) x 32 x 32

        self.convt10 = nn.ConvTranspose2d( self.nf * 4, self.nf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(self.nf * 2)
        self.act10 = nn.LeakyReLU(0.2, inplace=True)
        # サイズ：(self.nf*2) x 64 x 64

        self.convt11 = nn.ConvTranspose2d( self.nf * 2, self.nf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(self.nf)
        self.act11 = nn.LeakyReLU(0.2, inplace=True)
        # サイズ：(self.nf) x 128 x 128

        self.convt12 = nn.ConvTranspose2d( self.nf, self.nc, kernel_size=4, stride=2, padding=1, bias=False)

        # サイズ： (nc) x 256 x 256

    def forward(self, x):
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        
        conv2 = self.conv2(pool1)
        bn2 = self.bn2(conv2)
        act2 = self.act2(bn2)
        pool2 = self.pool2(act2)
        
        conv3 = self.conv3(pool2)
        bn3 = self.bn3(conv3)
        act3 = self.act3(bn3)
        pool3 = self.pool3(act3)
        
        conv4 = self.conv4(pool3)
        bn4 = self.bn4(conv4)
        act4 = self.act4(bn4)
        pool4 = self.pool4(act4)
        
        conv5 = self.conv5(pool4)
        bn5 = self.bn5(conv5)
        act5 = self.act5(bn5)
        pool5 = self.pool5(act5)
        
        conv6 = self.conv6(pool5)
        bn6 = self.bn6(conv6)
        act6 = self.act6(bn6)
        pool6 = self.pool6(act6)
        
        pool6 = pool6.view(pool6.size(0), -1)
        fc1 = self.fc1(pool6)
        fc2 =self.fc2(fc1)
        reshape = fc2.view((-1, 256, 4, 4))

        convt7 = self.convt7(reshape)
        bn7 = self.bn7(convt7)
        act7 = self.act7(bn7)
        
        convt8 = self.convt8(act7)
        bn8 = self.bn8(convt8)
        act8 = self.act8(bn8)
        
        convt9 = self.convt9(act8)
        bn9 = self.bn9(convt9)
        act9 = self.act9(bn9)
        
        convt10 = self.convt10(act9)
        bn10 = self.bn10(convt10)
        act10 = self.act10(bn10)
        
        convt11 = self.convt11(act10)
        bn11 = self.bn11(convt11)
        act11 = self.act11(bn11)
        
        convt12 = self.convt12(act11)
        act12 = torch.sigmoid(convt12)        
        
        x = act12        
        
        return x
"""