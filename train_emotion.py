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


def train_emotionnet(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(config.result_path):
        os.makedirs(config.result_path)
    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = TrainDataset(config.image_dir, config.csv_dir, config.landmark_dir, config.train_filelist,
                                 config.image_size, transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    test_dataset = TestDataset(config.image_dir, config.csv_dir, config.landmark_dir, config.test_filelist, config.image_size, transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    emotionnet = EmotionNet(pooling=config.pooling).to(device)

    if config.load_model:
        emotionnet.load_state_dict(torch.load(config.load_model))

    # Multi-GPU
    if config.device_ids:
        device_ids = [int(ids) for ids in config.device_ids.split(',')]
        emotionnet = torch.nn.DataParallel(emotionnet, device_ids=device_ids)

    optim = torch.optim.Adam(params=emotionnet.parameters(), lr=config.lr)

    if config.loss_type == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif config.loss_type == 'l2':
        criterion = torch.nn.MSELoss().to(device)
    elif config.loss_type == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss().to(device)
    else:
        raise NotImplementedError('Loss type [{:s}] is not supported'.format(config.loss_type))

    for epoch in range(1 + config.continue_train, config.epochs + 1):

        for i, data in enumerate((train_loader)):
            images = data['image'].to(device)
            au_regress = data['au_regress'].float().to(device)
            au_cls = data['au_cls'].float().to(device)

            au_feat = emotionnet(images)

            optim.zero_grad()
            loss = criterion(au_feat, au_regress)
            loss.backward()
            optim.step()

            if i % 50 == 0:
                print('Epoch: %d/%d | Step: %d/%d | Loss: %.4f' % (epoch, config.epochs, i, len(train_loader), loss.item()))

        emotionnet.eval()
        total_loss = 0
        for i, data in enumerate(test_loader):
            image = data['image'].to(device)
            au_regress = data['au_regress'].float().to(device)
            au_feat = emotionnet(image)
            loss = criterion(au_feat, au_regress)
            total_loss += loss.item()
        total_loss /= len(test_loader)
        print('Evaluation loss: %.4f' % total_loss)
        emotionnet.train()

        if epoch % 5 == 0:
            if config.device_ids:
                torch.save(emotionnet.module.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))
            else:
                torch.save(emotionnet.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))


def train_glemotionnet(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(config.result_path):
        os.makedirs(config.result_path)
    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = TrainDataset(config.image_dir, config.csv_dir, config.landmark_dir, config.train_filelist,
                                 config.image_size, transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    test_dataset = TestDataset(config.image_dir, config.csv_dir, config.landmark_dir, config.test_filelist, config.image_size, transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    emotionnet = GLEmotionnet(pooling=config.pooling).to(device)

    if config.load_model:
        emotionnet.load_state_dict(torch.load(config.load_model))

    # Multi-GPU
    if config.device_ids:
        device_ids = [int(ids) for ids in config.device_ids.split(',')]
        emotionnet = torch.nn.DataParallel(emotionnet, device_ids=device_ids)

    optim = torch.optim.Adam(params=emotionnet.parameters(), lr=config.lr)

    if config.loss_type == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif config.loss_type == 'l2':
        criterion = torch.nn.MSELoss().to(device)
    elif config.loss_type == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss().to(device)
    else:
        raise NotImplementedError('Loss type [{:s}] is not supported'.format(config.loss_type))

    for epoch in range(1 + config.continue_train, config.epochs + 1):

        for i, data in enumerate((train_loader)):
            images = data['image'].to(device)
            au_regress = data['au_regress'].float().to(device)
            au_cls = data['au_cls'].float().to(device)
            landmarks = data['landmark'].float().to(device)

            au_feat, landmark_feat = emotionnet(images)

            optim.zero_grad()
            au_loss = criterion(au_feat, au_regress)
            landmark_loss = config.l_landmark * criterion(landmark_feat, landmarks)
            loss = au_loss + landmark_loss
            loss.backward()
            optim.step()

            if i % 50 == 0:
                print('Epoch: %d/%d | Step: %d/%d | AU loss: %.5f  Landmark loss: %.5f' %
                      (epoch, config.epochs, i, len(train_loader), au_loss.item(), landmark_loss.item()))

            # if i % 1 == 0:
            #     image = (images.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255)
            #     image = np.ascontiguousarray(image, dtype=np.uint8)
            #     landmark = landmarks.detach().cpu().numpy()[0] * config.image_size
            #     for l in landmark:
            #         cv2.circle(image, (int(l[0]), int(l[1])), radius=1, color=(255, 0, 0), thickness=1)
            #     cv2.imwrite(join(config.result_path, 'epoch-%d-step-%d.png' % (epoch, i)), image)

        emotionnet.eval()
        total_au_loss = 0
        total_landmark_loss = 0
        for i, data in enumerate(test_loader):
            image = data['image'].to(device)
            au_regress = data['au_regress'].float().to(device)
            landmarks = data['landmark'].float().to(device)
            au_feat, landmark_feat = emotionnet(image)
            au_loss = criterion(au_feat, au_regress)
            landmark_loss = criterion(landmark_feat, landmarks)
            total_au_loss += au_loss.item()
            total_landmark_loss += landmark_loss.item()
        total_au_loss /= len(test_loader)
        total_landmark_loss /= len(test_loader)
        print('Evaluation AU loss: %.4f, Landmark loss: %.4f' % (total_au_loss, total_landmark_loss))
        emotionnet.train()

        if epoch % 5 == 0:
            if config.device_ids:
                torch.save(emotionnet.module.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))
            else:
                torch.save(emotionnet.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/clip-faces/')
    parser.add_argument('--csv_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/action-units/')
    parser.add_argument('--landmark_dir', type=str, default='/media/liuwq/data/Dataset/emotionnet/CK+/clip-landmarks/')
    parser.add_argument('--train_filelist', type=str, default='data/train_filelist.txt')
    parser.add_argument('--test_filelist', type=str, default='data/test_filelist.txt')
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--device_ids', type=str, default=None)

    parser.add_argument('--l_landmark', type=float, default=1.0)

    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--load_model', type=str, default=None)

    parser.add_argument('--use_model', type=str, default='emotionnet')

    config = parser.parse_args()

    if config.use_model == 'emotionnet':
        train_emotionnet(config)
    elif config.use_model == 'glemotionnet':
        train_glemotionnet(config)
