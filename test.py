from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import glob

from data_loader import Rescale
from data_loader import ToTensor
from data_loader import SalObjDataset

from model import DACNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name, pred, d_dir):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]       
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')

image_dir = ""
prediction_dir = ""
model_dir = ""

img_name_list = glob.glob(image_dir + '*.bmp')

test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([Rescale(256),ToTensor(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False)

net = DACNet(3,1)
net.load_state_dict(torch.load(model_dir))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()

for i_test, data_test in enumerate(test_salobj_dataloader):

	inputs_test = data_test['image']
	inputs_test = inputs_test.type(torch.FloatTensor).to(device)

	d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

	pred = d1[:,0,:,:]
	pred = normPRED(pred)

	save_output(img_name_list[i_test],pred,prediction_dir)

	del d1,d2,d3,d4,d5,d6,d7
