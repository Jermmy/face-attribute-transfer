import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, models

from model.loss import GramLoss
from model.vgg import VGG16

import cv2
import numpy as np
import os
from os.path import join, exists
import argparse

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def tensor2image(img):
    img = img.detach().cpu().numpy()[0].transpose((1, 2, 0))
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def train(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    vgg = VGG16().to(device)
    vgg.load_model(config.vgg16)

    # vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    content_img = cv2.imread(config.content_img)
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
    style_img = cv2.imread(config.style_img)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGRA2RGB)

    content_img = prep(content_img)
    style_img = prep(style_img)

    content_img = Variable(content_img.unsqueeze(0)).to(device)
    style_img = Variable(style_img.unsqueeze(0)).to(device)

    content_layers = config.content_layers.split(',')
    style_layers = config.style_layers.split(',')

    gramLoss = GramLoss().to(device)
    vggLoss = torch.nn.MSELoss().to(device)

    opt_img = Variable(torch.randn(content_img.size()).type_as(content_img.data), requires_grad=True)
    # opt_img = Variable(content_img.data.clone(), requires_grad=True).to(device)

    optim = torch.optim.LBFGS(params=[opt_img], lr=config.lr)

    for epoch in range(1, config.epochs + 1):

        def closure():
            optim.zero_grad()

            content_loss = 0
            style_loss = 0

            content_feat = vgg(content_img)
            style_feat = vgg(style_img)
            opt_feat = vgg(opt_img)
            for cl in content_layers:
                content_loss += vggLoss(opt_feat[cl], content_feat[cl].detach())
            content_loss = config.lc * content_loss

            for sl in style_layers:
                style_loss += gramLoss(opt_feat[sl], style_feat[sl].detach())
            style_loss = config.ls * style_loss

            loss = content_loss + style_loss
            loss.backward()

            print('Epoch: %d | content loss: %.4f, style loss: %.4f' % (epoch, content_loss.item(), style_loss.item()))
            if epoch % 10 == 0:
                cv2.imwrite(join(config.result_path, 'epoch-%d.jpg' % epoch), tensor2image(opt_img))

            return loss

        optim.step(closure)

        # print('Epoch: %d | content loss: %.4f, style loss: %.4f' % (epoch, content_loss.item(), style_loss.item()))

        # if epoch % 10 == 0:
        #     cv2.imwrite(join(config.result_path, 'epoch-%d.jpg' % epoch), tensor2image(opt_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg16', type=str, default='pretrain/vgg16-397923af.pth')
    parser.add_argument('--content_img', type=str, default='data/Tuebingen_Neckarfront.jpg')
    parser.add_argument('--style_img', type=str, default='data/vangogh_starry_night.jpg')

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lc', type=float, default=1.)
    parser.add_argument('--ls', type=float, default=0.)

    parser.add_argument('--content_layers', type=str, default='r42')
    parser.add_argument('--style_layers', type=str, default='c22,c31')
    parser.add_argument('--result_path', type=str, default='result')

    config = parser.parse_args()

    train(config)




