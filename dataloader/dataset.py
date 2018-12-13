import torch.utils.data as data

from os.path import join
import numpy as np
import cv2


class TrainDataset(data.Dataset):

    def __init__(self, image_dir, csv_dir, landmark_dir, filelist, image_size, transform):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.landmark_dir = landmark_dir
        self.image_size = image_size
        self.image_files = []
        with open(filelist, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.image_files += [line]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = join(self.image_dir, self.image_files[idx])
        csv_file = join(self.csv_dir, self.image_files[idx].replace('png', 'csv'))
        landmark_file = join(self.landmark_dir,
                             self.image_files[idx].replace('.png', '_landmarks.txt'))

        image = cv2.imread(image_file)
        height, width = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        au = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        au_regress = au[5:22]
        au_cls = au[22:]

        landmarks = []
        scale_height = height / self.image_size
        scale_width = width / self.image_size
        with open(landmark_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                landmarks += [[float(line[0]) / scale_width, float(line[1]) / scale_height]]
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape((68 * 2,))

        landmarks /= self.image_size

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'au_regress': au_regress, 'au_cls': au_cls, 'landmark': landmarks}
        return sample


class TestDataset(TrainDataset):

    def __init__(self, image_dir, csv_dir, landmark_dir, filelist, image_size, transform):
        super(TestDataset, self).__init__(image_dir, csv_dir, landmark_dir, filelist, image_size, transform)
