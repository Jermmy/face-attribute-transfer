import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from dataloader.dataset import TrainDataset, TestDataset
from model.emotionnet import EmotionNet, GLEmotionnet

import cv2
import numpy as np
from tqdm import tqdm
import os
from os.path import join, exists
import argparse


def test_glemotionnet(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    emotionnet = GLEmotionnet(pooling=config.pooling).to(device)

    if config.load_model:
        emotionnet.load_state_dict(torch.load(config.load_model))
    emotionnet.eval()

    image = cv2.imread(config.test_image)
    image = cv2.resize(image, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    au = np.loadtxt(config.csv_file, delimiter=',', skiprows=1)
    au_regress = au[5:22]
    au_regress = torch.from_numpy(au_regress).unsqueeze(0).float().to(device)

    if config.loss_type == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif config.loss_type == 'l2':
        criterion = torch.nn.MSELoss().to(device)
    elif config.loss_type == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss().to(device)
    else:
        raise NotImplementedError('Loss type [{:s}] is not supported'.format(config.loss_type))

    au_feat, _ = emotionnet(image)

    au_loss = criterion(au_feat, au_regress)

    print('GT: %s\nPredict:%s\n' % (str(au_regress), str(au_feat)))

    print('AU Loss: %.6f' % au_loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/clip-faces/')
    # parser.add_argument('--csv_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/action-units/')
    # parser.add_argument('--landmark_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/clip-landmarks/')
    # parser.add_argument('--train_filelist', type=str, default='data/train_filelist.txt')
    # parser.add_argument('--test_filelist', type=str, default='data/test_filelist.txt')

    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--test_image', type=str, default='S010_001_00000001.png')
    parser.add_argument('--csv_file', type=str, default='S010_001_00000001.csv')

    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--loss_type', type=str, default='smoothl1')
    parser.add_argument('--load_model', type=str, default='ckpt/glemotionnet/lr_1e-4_pooling_avg_smoothl1/epoch-20.pkl')

    parser.add_argument('--use_model', type=str, default='glemotionnet')

    config = parser.parse_args()

    if config.use_model == 'glemotionnet':
        test_glemotionnet(config)
