import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from dataloader.dataset import TrainDataset, TestDataset
from model.emotionnet import EmotionNet

import cv2
import numpy as np
import os
from os.path import join, exists
import argparse


def train(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(config.result_path):
        os.makedirs(config.result_path)
    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = TrainDataset(config.image_dir, config.csv_dir, config.train_filelist, config.image_size,
                                 transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)\

    test_dataset = TestDataset(config.image_dir, config.csv_dir, config.test_filelist, config.image_size, transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    emotionnet = EmotionNet(pooling=config.pooling).to(device)

    if config.load_model:
        emotionnet.load_state_dict(torch.load(config.load_model))

    optim = torch.optim.Adam(params=emotionnet.parameters(), lr=config.lr)

    if config.loss_type == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif config.loss_type == 'mse':
        criterion = torch.nn.MSELoss().to(device)
    elif config.loss_type == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss().to(device)
    else:
        raise NotImplementedError('Loss type [{:s}] is not supported'.format(config.loss_type))

    for epoch in range(1 + config.continue_train, config.epochs + 1):

        for i, data in enumerate(train_loader):
            image = data['image'].to(device)
            au = data['au'].to(device)

            au_feat = emotionnet(image)

            optim.zero_grad()
            loss = criterion(au_feat, au)
            loss.backward()
            optim.step()

            if i % 500 == 0:
                print('Epoch: %d/%d  loss: %.4f' % (epoch, len(config.epochs), loss.item()))

        emotionnet.eval()
        total_loss = 0
        for i, data in enumerate(test_loader):
            image = data['image'].to(device)
            au = data['au'].to(device)
            au_feat = emotionnet(image)
            loss = criterion(au_feat, au)
            total_loss += loss.item()
        total_loss /= len(test_loader)
        print('Evaluation loss: %.4f' % total_loss)
        emotionnet.train()

        if epoch % 5 == 0:
            torch.save(emotionnet.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--csv_dir', type=str, default='')
    parser.add_argument('--train_filelist', type=str, default='data/train_filelist.txt')
    parser.add_argument('--test_filelist', type=str, default='data/test_filelist.txt')
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--continue_train', type=int, default=0)

    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--load_model', type=str, default=None)

    config = parser.parse_args()
    train(config)
