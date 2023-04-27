import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

batch_w = 256
batch_h = 256


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count
random.seed(1143)
def populate_list(images_path):

	image_list = glob.glob(images_path +"/**/"+"*.jpg",recursive=True)

	random.shuffle(image_list)

	return image_list


class paired_data_loader(data.Dataset):

	def __init__(self, images_path):
		self.data_list = populate_list(os.path.join(images_path,"low"))
		self.label_list = [path.replace("low","high") for path in self.data_list]
		self.size = 256
		print("Total data:"+str(len(self.data_list))+"Total label:"+str(len(self.label_list)))

	def __getitem__(self, index):
		data_path = self.data_list[index]
		label_path = self.label_list[index]
		data = Image.open(data_path)
		label = Image.open(label_path)
		data = data.resize((self.size, self.size), Image.ANTIALIAS)
		label =  label.resize((self.size, self.size), Image.ANTIALIAS)
		data = (np.asarray(data) / 255.0)
		label = (np.asarray(label) / 255.0)
		data = torch.from_numpy(data).float()
		label = torch.from_numpy(label).float()

		return data.permute(2, 0, 1), label.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

