from typing_extensions import SupportsIndex
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from torchvision import transforms
dir = os.path.dirname(__file__)

class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir = None,  split='train', transform=None):
        super().__init__()
        self.data_dir = dir+f"/../cifar10/{split}" if data_dir is None else data_dir+"/"+split
        self.data = []
        self.targets = []
        self.classmap = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                         'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        self._load_data()
        self.transform = None # janky code needed to calculate mean and std as a part of the class
        self.mean, self.std = self._get_mean_std()
        # append mean/std scaling transform
        if transform is not None:
            transform = torch.nn.Sequential(transform, transforms.Normalize(self.mean, self.std))
        else:
            transform = transforms.Normalize(self.mean, self.std)
        self.transform = transform

    def _get_mean_std(self):
        # compute mean and std of the dataset: this should be done only once for the whole dataset
        try: # find cifar_10_mean_std.py in the same folder
            mean, std = np.load(dir+'/cifar10_mean_std.npy')
            return mean, std
        except: # if not found, calculate mean and std and save it to the file
            mean = np.zeros(3, dtype=np.float64)
            std = np.zeros(3, dtype=np.float64)
            for img, _ in self:
                mean += img.mean(axis=(1, 2)).numpy()
                std += img.std(axis=(1, 2)).numpy()
            mean /= len(self)
            std /= len(self)
            np.save(dir+'/cifar10_mean_std.npy', [mean, std])
            return mean, std
    
    def _load_data(self):
        # walk the data dir and store all the images in a list
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".png"):
                    self.data.append(os.path.join(root, file))
                    self.targets.append(self.classmap[root.split('/')[-1]])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = plt.imread(self.data[idx])
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC -> CHW, required as torch is channel major
        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]
    
    def get_class_name(self, idx):
        '''
        [general utility func] Returns the class name for a given class index
        :param idx: the class index
        :return: the class name
        '''
        return list(self.classmap.keys())[idx]