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

    content_image = cv2.imread(config.content_image)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    style_image = cv2.imread(config.style_image)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGRA2RGB)

    content_image = prep(content_image)
    style_image = prep(style_image)

    content_image = Variable(content_image.unsqueeze(0)).to(device)
    style_image = Variable(style_image.unsqueeze(0)).to(device)

    content_layers = config.content_layers.split(',')
    style_layers = config.style_layers.split(',')




