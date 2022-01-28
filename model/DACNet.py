import torch
import torch.nn as nn
from torchvision import models

from .resnet_model import *

class DACNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(DACNet,self).__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------
        self.pooling2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.pooling4 = nn.MaxPool2d(4,4,ceil_mode=True)

        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #256
        #stage 2
        self.encoder2 = resnet.layer2 #128
        #stage 3
        self.encoder3 = resnet.layer3 #64
        #stage 4
        self.encoder4 = resnet.layer4 #32

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #16

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #8

        #cascade
        self.convc1 = nn.Conv2d(1024,512,3,padding=1)
        self.convc2 = nn.Conv2d(1536,512,3,padding=1)

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1)
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1)
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1025,512,3,padding=1)
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(514,256,3,padding=1)
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(259,128,3,padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(132,64,3,padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear',align_corners = True)
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear',align_corners = True)
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners = True)
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners = True)
        self.upscore2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners = True)

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

    def forward(self,x):

        hx = x
        hx1 = x
        hx2 = x

        ## -------------Encoder1-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Encoder2-------------
        hx1 = self.pooling2(hx1) # 128
        hx1 = self.inconv(hx1)
        hx1 = self.inbn(hx1)
        hx1 = self.inrelu(hx1)

        h1_1 = self.encoder1(hx1) # 128
        h1_2 = self.encoder2(h1_1) # 64
        h1_3 = self.encoder3(h1_2) # 32
        h1_4 = self.encoder4(h1_3) # 16

        hx1 = self.pool4(h1_4) # 8 

        hx1 = self.resb5_1(hx1)
        hx1 = self.resb5_2(hx1)
        h1_5 = self.resb5_3(hx1) #channel=512
        
        ## -------------Encoder3-------------
        hx2 = self.pooling4(hx2) # 64
        hx2 = self.inconv(hx2)
        hx2 = self.inbn(hx2)
        hx2 = self.inrelu(hx2)

        h2_1 = self.encoder1(hx2) # 64
        h2_2 = self.encoder2(h2_1) # 32
        h2_3 = self.encoder3(h2_2) # 16
        h2_4 = self.encoder4(h2_3) # 8 channel=512

        ## -------------cascade-------------
        c1 = self.convc1(torch.cat((h2_4,h1_5),1))
        h6 = self.convc2(torch.cat((h6,c1,h2_4),1))
        
        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------
        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16
        hd6 = self.outconv6(hd6)
        hd6_1 = self.upscore3(hd6)
        hd6_2 = self.upscore4(hd6)
        hd6_3 = self.upscore5(hd6)
        d6 = self.upscore6(hd6)# 8->256

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32
        hd5 = self.outconv5(hd5)
        hd5_1 = self.upscore3(hd5)
        hd5_2 = self.upscore4(hd5)
        d5 = self.upscore5(hd5)# 16->256

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4,hd6_1),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64
        hd4 = self.outconv4(hd4)
        hd4_1 = self.upscore3(hd4)
        d4 = self.upscore4(hd4)# 32->256

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3,hd6_2,hd5_1),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128
        hd3 = self.outconv3(hd3)
        d3 = self.upscore3(hd3) # 64->256

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2,hd6_3,hd5_2,hd4_1),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256
        hd2 = self.outconv2(hd2)
        d2 = self.upscore2(hd2) # 128->256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1,d6,d5,d4,d3),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))
        d1 = self.outconv1(hd1) # 256

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        return torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6), torch.sigmoid(db)
