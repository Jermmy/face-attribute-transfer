import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(torch.nn.Module):
    def __init__(self):
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

    def forward(self, X):
        features = {}
        features['conv1_1'] = self.conv1_1(X)
        features['relu1_1'] = F.relu(features['conv1_1'])
        features['conv1_2'] = self.conv1_2(features['relu1_1'])
        features['relu1_2'] = F.relu(features['conv1_2'])
        features['maxpool1_2'] = F.max_pool2d(features['relu1_2'], kernel_size=2, stride=2)

        features['conv2_1'] = self.conv2_1(features['maxpool1_2'])
        features['relu2_1'] = F.relu(features['conv2_1'])
        features['conv2_2'] = self.conv2_2(features['relu2_1'])
        features['relu2_2'] = F.relu(features['conv2_2'])
        features['maxpool2_2'] = F.max_pool2d(features['relu2_2'], kernel_size=2, stride=2)

        features['conv3_1'] = self.conv3_1(features['maxpool2_2'])
        features['relu3_1'] = F.relu(features['conv3_1'])
        features['conv3_2'] = self.conv3_2(features['relu3_1'])
        features['relu3_2'] = F.relu(features['conv3_2'])
        features['conv3_3'] = self.conv3_3(features['relu3_2'])
        features['relu3_3'] = F.relu(features['conv3_3'])
        features['maxpool3_3'] = F.max_pool2d(features['relu3_3'], kernel_size=2, stride=2)

        features['conv4_1'] = self.conv4_1(features['maxpool3_3'])
        features['relu4_1'] = F.relu(features['conv4_1'])
        features['conv4_2'] = self.conv4_2(features['relu4_1'])
        features['relu4_2'] = F.relu(features['conv4_2'])
        features['conv4_3'] = self.conv4_3(features['relu4_2'])
        features['relu4_3'] = F.relu(features['conv4_3'])

        return features

    def load_model(self, model_file):
        vgg16_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)
        vgg16_keys = vgg16_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg16_keys, pretrained_keys):
            vgg16_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg16_dict)

