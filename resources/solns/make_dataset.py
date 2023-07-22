import torch
import numpy as np
import os
dir = os.path.dirname(__file__)

class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir = None,  split='train', transform=None):
        super().__init__()
        self.data_dir = dir+f"/../cifar10/{split}" if data_dir is None else data_dir+"/"+split
        self.transform = transform
        self.data = []
        self.targets = []
        self._load_data()

    def _load_data(self):
        # walk the data dir and count the number of files
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".png"):
                    data = np.load(os.path.join(root, file))
                    self.data.append(data)
                    self.targets.append(int(file.split("_")[0]))