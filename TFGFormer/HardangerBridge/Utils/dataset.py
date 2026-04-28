import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, feature_path, label_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))   
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))     

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])
        feature_data = torch.from_numpy(feature_data)  
        label_data = torch.from_numpy(label_data)                           
        feature_data.unsqueeze_(0)
        label_data.unsqueeze_(0)
        return feature_data, label_data