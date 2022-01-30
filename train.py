import glob
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from data_loader import Rescale
from data_loader import ToTensor
from data_loader import SalObjDataset

from model import DACNet

import pytorch_ssim
import pytorch_iou

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def hybrid_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def multi_loss(d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss1 = hybrid_loss(d1,labels_v)
	loss2 = hybrid_loss(d2,labels_v)
	loss3 = hybrid_loss(d3,labels_v)
	loss4 = hybrid_loss(d4,labels_v)
	loss5 = hybrid_loss(d5,labels_v)
	loss6 = hybrid_loss(d6,labels_v)
	loss7 = hybrid_loss(d7,labels_v)

	loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

	return loss

tra_img_dir = ""
tra_lbl_dir = ""

image_ext = '.bmp'
label_ext = '.png'

model_dir = ""

epoch_num = 600
batch_size_train = 10
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(tra_img_dir + '*' + image_ext)
tra_lbl_name_list = []

for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]          
        imgIdx = img_name.split(".")[0]
        tra_lbl_name_list.append(tra_lbl_dir + imgIdx + label_ext)

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        Rescale(256),
        ToTensor(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True,num_workers=4)

net = DACNet(3,1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

ite_num = 0
running_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs_v = inputs.type(torch.FloatTensor).to(device)
        labels_v = labels.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss = multi_loss(d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        del d1, d2, d3, d4, d5, d6, d7, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))
    if (epoch+1) % 50 == 0:       
        torch.save(net.state_dict(), model_dir + "DACNet_epoch_%d.pth" % (epoch+1))
    running_loss = 0.0
    net.train()
    ite_num4val = 0
