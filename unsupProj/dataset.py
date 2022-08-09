'''
    Creating dataset class for cropped camera trap images.
    2022 Peggy Bevan 
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDataset(Dataset):


	def __init__(self, cfg, img_list_path, transform):
	# data_root = base directory for imgs - useful for moving between local and vm
		txt_file = open(img_list_path, "r")
		self.img_list = txt_file.readlines()
		self.base = cfg['basename']
		self.transform = Compose([Resize((cfg['image_size'])),ToTensor()])
		 # Transforms. For now, resizes image to dims needed for Res50 and converts to tensor
	def __len__(self):
		return len(self.img_list)
		'''
		    Returns the length of the dataset.
		'''
	def __getitem__(self, idx):
		'''
		Returns a single data point at given idx.
		Here's where we actually load the image.
		'''
		image_path = os.path.join(self.base, self.img_list[idx])
		# load image
		img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

		# transform: see lines above where we define our transformations
		img_tensor = self.transform(img)

		return img_tensor
