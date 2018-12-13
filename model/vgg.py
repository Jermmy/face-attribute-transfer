import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models

class VGG16(torch.nn.Module):

    def __init__(self, pooling='max'):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('Pooling type [{:s}] is not supported'.format(pooling))

    def forward(self, X):
        features = {}
        features['c11'] = self.conv1_1(X)
        features['r11'] = F.relu(features['c11'])
        features['c12'] = self.conv1_2(features['r11'])
        features['r12'] = F.relu(features['c12'])
        features['p12'] = self.pooling(features['r12'])

        features['c21'] = self.conv2_1(features['p12'])
        features['r21'] = F.relu(features['c21'])
        features['c22'] = self.conv2_2(features['r21'])
        features['r22'] = F.relu(features['c22'])
        features['p22'] = self.pooling(features['r22'])

        features['c31'] = self.conv3_1(features['p22'])
        features['r31'] = F.relu(features['c31'])
        features['c32'] = self.conv3_2(features['r31'])
        features['r32'] = F.relu(features['c32'])
        features['c33'] = self.conv3_3(features['r32'])
        features['r33'] = F.relu(features['c33'])
        features['p33'] = self.pooling(features['r33'])

        features['c41'] = self.conv4_1(features['p33'])
        features['r41'] = F.relu(features['c41'])
        features['c42'] = self.conv4_2(features['r41'])
        features['r42'] = F.relu(features['c42'])
        features['c43'] = self.conv4_3(features['r42'])
        features['r43'] = F.relu(features['c43'])

        return features

    def forward_feat(self, X):
        return self.forward(X)

    def load_model(self, model_file):
        vgg16_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)
        vgg16_keys = vgg16_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg16_keys, pretrained_keys):
            vgg16_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg16_dict)

