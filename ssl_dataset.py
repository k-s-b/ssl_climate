import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from imresize import imresize
from gkernel import generate_kernel
from PIL import Image

import random
from cutmix.utils import onehot, rand_bbox
import cv2

class HRClimateDataset(Dataset):
    """Use it for downscaling for ZSSR type low resolution dataset."""
    def __init__(self, lr_dir=None, hr_dir=None, season_dir=None):
        self.hr_data = np.load(hr_dir)[:1]
        # self.season = np.load(season_dir)

    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, idx):
        hr_sample  = self.hr_data[idx]
        hr_sample = hr_sample.transpose((2, 0, 1))
        hr_sample = torch.from_numpy(hr_sample).to(torch.float32)

        # season = self.season[idx]

        return hr_sample

class ZSSRTypeDataset(Dataset):
    """Use it for downscaling for ZSSR type low resolution dataset."""
    def __init__(self, dataset, scale_factor = 2., gauss_noise=None, transform=None, topo_transform = None, data_aug=None):
        self.hr_data = dataset
        self.transform = transform
        self.data_aug = data_aug
        self.topo_transform = topo_transform
        self.scale_factor = scale_factor
        self.gauss_noise = gauss_noise


    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, idx):
        hr_sample  = self.hr_data[idx] # 0 because hr sample is returned from previous dataset class; self.hr_data[idx] is a tuple of hr data, season

        if(self.gauss_noise):

            hr_sample = return_sample(self.scale_factor, hr_sample.cpu().numpy())
            hr_sample = torch.from_numpy(hr_sample).to(torch.float32)

        hr_sample = hr_sample.unsqueeze(0)

        if self.data_aug:
            hr_sample = self.data_aug(hr_sample)


        lr_father = F.interpolate(hr_sample, (213, 321), mode= 'bicubic') #upsize father to original (the size ot whihc we want to interpolate) size
        lr_son = F.interpolate(lr_father, scale_factor = (1./self.scale_factor), mode= 'bicubic') #downsize upsized father
        lr_sample = F.interpolate(lr_son, (lr_father.shape[2], lr_father.shape[3]), mode= 'bicubic').squeeze(0) #upsize son; interpolate LR son to LR father, whihc now becomes a LR sample to be used
        hr_sample = lr_father.to(torch.float32).squeeze(0) #LR father now becomes the HR sample to be used


        season = 1

        if self.transform:
            lr_sample = self.transform(lr_sample)
            hr_sample = self.transform(hr_sample)

        return lr_sample, hr_sample, season
