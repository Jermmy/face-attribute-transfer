import torch.utils.data as data

import os
from os.path import join, exists
import numpy as np
import cv2


class TrainDataset(data.Dataset):

    def __init__(self, image_dir, csv_dir, image_size, transform):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('png')]
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith('csv')]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.imread(join(self.image_dir, self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        au = np.loadtxt(join(self.csv_dir, self.csv_files[idx]), delimiter=',', skiprows=1)
        au = au[2: 19]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'au': au}
        return sample
