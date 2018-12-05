import torch.utils.data as data

from os.path import join
import numpy as np
import cv2


class TrainDataset(data.Dataset):

    def __init__(self, image_dir, csv_dir, filelist, image_size, transform):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.image_size = image_size
        self.train_data = []
        with open(filelist, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.train_data += [line]

        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image_file = join(self.image_dir, self.train_data[idx][0])
        csv_file = join(self.csv_dir, self.train_data[idx][1])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        au = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        au = au[2: 19]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'au': au}
        return sample


class TestDataset(TrainDataset):

    def __init__(self, image_dir, csv_dir, filelist, image_size, transform):
        super(TestDataset, self).__init__(image_dir, csv_dir, filelist, image_size, transform)
