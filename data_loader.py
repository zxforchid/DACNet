import torch
from skimage import io, transform, color
import numpy as np
from torch.utils.data import Dataset

class Rescale(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
		lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

		return {'image': img, 'label': lbl}

class ToTensor(object):
	def __init__(self, flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label = sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if np.max(label) < 1e-6:
			label = label
		else:
			label = label/np.max(label)

		if self.flag == 2:
			tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
			tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
			if image.shape[2] == 1:
				tmpImgt[:, :, 0] = image[:, :, 0]
				tmpImgt[:, :, 1] = image[:, :, 0]
				tmpImgt[:, :, 2] = image[:, :, 0]
			else:
				tmpImgt = image

			tmpImgtl = color.rgb2lab(tmpImgt)

			tmpImg[:, :, 0] = (tmpImgt[:, :, 0]-np.min(tmpImgt[:, :, 0]))/(np.max(tmpImgt[:, :, 0])-np.min(tmpImgt[:, :, 0]))
			tmpImg[:, :, 1] = (tmpImgt[:, :, 1]-np.min(tmpImgt[:, :, 1]))/(np.max(tmpImgt[:, :, 1])-np.min(tmpImgt[:, :, 1]))
			tmpImg[:, :, 2] = (tmpImgt[:, :, 2]-np.min(tmpImgt[:, :, 2]))/(np.max(tmpImgt[:, :, 2])-np.min(tmpImgt[:, :, 2]))
			tmpImg[:, :, 3] = (tmpImgtl[:, :, 0]-np.min(tmpImgtl[:, :, 0]))/(np.max(tmpImgtl[:, :, 0])-np.min(tmpImgtl[:, :, 0]))
			tmpImg[:, :, 4] = (tmpImgtl[:, :, 1]-np.min(tmpImgtl[:, :, 1]))/(np.max(tmpImgtl[:, :, 1])-np.min(tmpImgtl[:, :, 1]))
			tmpImg[:, :, 5] = (tmpImgtl[:, :, 2]-np.min(tmpImgtl[:, :, 2]))/(np.max(tmpImgtl[:, :, 2])-np.min(tmpImgtl[:, :, 2]))

			tmpImg[:, :, 0] = (tmpImg[:, :, 0]-np.mean(tmpImg[:, :, 0]))/np.std(tmpImg[:, :, 0])
			tmpImg[:, :, 1] = (tmpImg[:, :, 1]-np.mean(tmpImg[:, :, 1]))/np.std(tmpImg[:, :, 1])
			tmpImg[:, :, 2] = (tmpImg[:, :, 2]-np.mean(tmpImg[:, :, 2]))/np.std(tmpImg[:, :, 2])
			tmpImg[:, :, 3] = (tmpImg[:, :, 3]-np.mean(tmpImg[:, :, 3]))/np.std(tmpImg[:, :, 3])
			tmpImg[:, :, 4] = (tmpImg[:, :, 4]-np.mean(tmpImg[:, :, 4]))/np.std(tmpImg[:, :, 4])
			tmpImg[:, :, 5] = (tmpImg[:, :, 5]-np.mean(tmpImg[:, :, 5]))/np.std(tmpImg[:, :, 5])

		elif self.flag == 1:                                            
			tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

			if image.shape[2] == 1:
				tmpImg[:, :, 0] = image[:, :, 0]
				tmpImg[:, :, 1] = image[:, :, 0]
				tmpImg[:, :, 2] = image[:, :, 0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			tmpImg[:, :, 0] = (tmpImg[:, :, 0]-np.min(tmpImg[:, :, 0]))/(np.max(tmpImg[:, :, 0])-np.min(tmpImg[:, :, 0]))
			tmpImg[:, :, 1] = (tmpImg[:, :, 1]-np.min(tmpImg[:, :, 1]))/(np.max(tmpImg[:, :, 1])-np.min(tmpImg[:, :, 1]))
			tmpImg[:, :, 2] = (tmpImg[:, :, 2]-np.min(tmpImg[:, :, 2]))/(np.max(tmpImg[:, :, 2])-np.min(tmpImg[:, :, 2]))
			tmpImg[:, :, 0] = (tmpImg[:, :, 0]-np.mean(tmpImg[:, :, 0]))/np.std(tmpImg[:, :, 0])
			tmpImg[:, :, 1] = (tmpImg[:, :, 1]-np.mean(tmpImg[:, :, 1]))/np.std(tmpImg[:, :, 1])
			tmpImg[:, :, 2] = (tmpImg[:, :, 2]-np.mean(tmpImg[:, :, 2]))/np.std(tmpImg[:, :, 2])

		else:                        
			tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
			image = image/np.max(image)
			if image.shape[2] == 1:
				tmpImg[:, :, 0] = (image[:, :, 0]-0.4669)/0.2437
				tmpImg[:, :, 1] = (image[:, :, 0]-0.4669)/0.2437
				tmpImg[:, :, 2] = (image[:, :, 0]-0.4669)/0.2437
			else:
				tmpImg[:, :, 0] = (image[:, :, 0]-0.4669)/0.2437
				tmpImg[:, :, 1] = (image[:, :, 1]-0.4669)/0.2437
				tmpImg[:, :, 2] = (image[:, :, 2]-0.4669)/0.2437

		tmpLbl[:, :, 0] = label[:, :, 0]

		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		image = io.imread(self.image_name_list[idx])

		if 0 == len(self.label_name_list):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if 3 == len(label_3.shape):
			label = label_3[:, :, 0]
		elif 2 == len(label_3.shape):
			label = label_3

		if 3 == len(image.shape) and 2 == len(label.shape):
			label = label[:, :, np.newaxis]
		elif 2 == len(image.shape) and 2 == len(label.shape):
			image = image[:, :, np.newaxis]
			label = label[:, :, np.newaxis]

		sample = {'image': image, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample
