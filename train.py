import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, models
from tensorboardX import SummaryWriter

from model.loss import GramLoss, CXLoss
from model.vgg import VGG16
from model.vggface import VGGFace
from model.emotionnet import EmotionNet, GLEmotionnet

import cv2
import numpy as np
import os
from os.path import join, exists
import argparse
import shutil

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def tensor2image(img, postFunc=None):
    # img = img.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img = img * std + mean
    # img = np.clip(img, 0, 1)
    # img = (img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return img
    img = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    if postFunc:
        img = postFunc(img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def train(config):
    print(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.device_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    writer = SummaryWriter(config.result_path)

    shutil.copy(config.content_img, join(config.result_path, config.content_img.split('/')[-1]))
    shutil.copy(config.style_img, join(config.result_path, config.style_img.split('/')[-1]))

    if config.network == 'vgg':
        prepFunc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        postFunc = lambda x: x * np.array(std) + np.array(mean)

        network = VGG16(pooling=config.pooling).to(device)
        network.load_model(config.ckpt_path)
    elif config.network == 'emotionnet':
        prepFunc = transforms.Compose([
            transforms.ToTensor()
        ])
        postFunc = None
        network = EmotionNet(pooling=config.pooling).to(device)
        network.load_state_dict(torch.load(config.ckpt_path))
    elif config.network == 'glemotionnet':
        prepFunc = transforms.Compose([
            transforms.ToTensor()
        ])
        postFunc = None
        network = GLEmotionnet(pooling=config.pooling).to(device)
        network.load_state_dict(torch.load(config.ckpt_path))
    elif config.network == 'vggface':
        prepFunc = transforms.Compose([
            transforms.ToTensor()
        ])
        postFunc = None
        network = VGGFace().to(device)
        network.load_model(config.ckpt_path)

    network.eval()

    content_img = cv2.imread(config.content_img)
    content_img = cv2.resize(content_img, (160, 160), interpolation=cv2.INTER_LINEAR)
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
    style_img = cv2.imread(config.style_img)
    style_img = cv2.resize(style_img, (160, 160), interpolation=cv2.INTER_LINEAR)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGRA2RGB)

    content_img = prepFunc(content_img)
    style_img = prepFunc(style_img)

    content_img = Variable(content_img.unsqueeze(0)).to(device)
    style_img = Variable(style_img.unsqueeze(0)).to(device)

    content_layers = config.content_layers.split(',')
    style_layers = config.style_layers.split(',')

    print(content_layers)
    print(style_layers)

    # gramLoss = GramLoss().to(device)
    if config.emotion_loss == 'cxloss':
        emotionLoss = CXLoss().to(device)
    elif config.emotion_loss == 'l2':
        emotionLoss = torch.nn.MSELoss().to(device)
    contentLoss = torch.nn.MSELoss().to(device)

    if config.init == 'noise':
        opt_img = Variable(torch.randn(content_img.size()).type_as(content_img.data), requires_grad=True).to(device)
    elif config.init == 'content':
        opt_img = Variable(content_img.data.clone(), requires_grad=True).to(device)

    lr = config.lr
    # optim = torch.optim.LBFGS(params=[opt_img], lr=lr)
    optim = torch.optim.Adam(params=[opt_img], lr=lr)

    for epoch in range(1, config.epochs + 1):

        def closure():
            optim.zero_grad()

            content_loss = 0
            style_loss = 0

            content_feat = network.forward_feat(content_img)
            style_feat = network.forward_feat(style_img)
            opt_feat = network.forward_feat(opt_img)
            for cl in content_layers:
                content_loss += contentLoss(opt_feat[cl], content_feat[cl].detach())
            content_loss = config.lc * content_loss

            for sl in style_layers:
                style_loss += emotionLoss(opt_feat[sl], style_feat[sl].detach())
            style_loss = config.ls * style_loss

            loss = content_loss + style_loss
            loss.backward()

            if (epoch - 1) % 100 == 0:
                # for sl in style_layers:
                #     print("sl: %s\nstyle_feat: %s\nopt_feat: %s\n\n" % (sl, style_feat[sl].detach().cpu().numpy(),
                #                                                     opt_feat[sl].detach().cpu().numpy()))
                print('Epoch: %d | content loss: %.6f, style loss: %.6f' % (
                epoch, content_loss.item(), style_loss.item()))
                cv2.imwrite(join(config.result_path, 'epoch-%d.jpg' % epoch), tensor2image(opt_img, postFunc))
                writer.add_scalars('loss', {'content loss': content_loss.item(),
                                            'emotion loss': style_loss.item()}, epoch)
            return loss

        # optim.step(closure)

        closure()
        optim.step()

        if epoch % 1000 == 0:
            lr /= 5
            # optim = torch.optim.LBFGS(params=[opt_img], lr=lr)
            optim = torch.optim.Adam(params=[opt_img], lr=lr)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='pretrain/vgg16-397923af.pth')
    parser.add_argument('--content_img', type=str, default='data/Tuebingen_Neckarfront.jpg')
    parser.add_argument('--style_img', type=str, default='data/vangogh_starry_night.jpg')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lc', type=float, default=1.)
    parser.add_argument('--ls', type=float, default=100.)
    parser.add_argument('--pooling', type=str, default='max')
    parser.add_argument('--emotion_loss', type=str, default='cxloss')

    parser.add_argument('--content_layers', type=str, default='r42')
    parser.add_argument('--style_layers', type=str, default='c22,c31')
    parser.add_argument('--result_path', type=str, default='result')

    parser.add_argument('--network', type=str, default='vgg')
    parser.add_argument('--init', type=str, default='content')
    parser.add_argument('--device_id', type=str, default='0')

    config = parser.parse_args()

    train(config)




