import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, models

from model.loss import GramLoss
from model.vgg import VGG16

import cv2
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def tensor2image(img):
    img = img.detach().cpu().numpy()[0].transpose((1, 2, 0))
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = img * 255
    return img


def train(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    vgg = VGG16().to(device)
    vgg.load_model(config.vgg16)

    vgg.eval()

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

    opt_img = Variable(content_img.data.clone(), required_grad=True)

    optim = torch.optim.Adam(params=opt_img, lr=config.lr)

    for epoch in range(config.epochs):
        optim.zero_grad()

        content_loss = 0
        style_loss = 0

        content_feat = vgg(content_img)
        style_feat = vgg(style_img)
        opt_feat = vgg(opt_img)
        for cl in content_layers:
            content_loss += config.lc * vggLoss(content_feat[cl].detach(), opt_feat[cl])

        for sl in style_layers:
            style_loss += config.ls * gramLoss(style_feat[sl].detach(), opt_feat[sl])

        loss = content_loss + style_loss
        loss.backward()

        optim.step()

        print('content loss: %.4f, style loss: %.4f' % (style_loss.item(), content_loss.item()))








