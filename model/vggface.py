import torch.nn as nn
import torch
import torch.nn.functional as F


meanvgg = [129.1863, 104.7624, 93.5940]


class VGGFace(nn.Module):

    def __init__(self):
        '''
        VGG Face model, only contains layer 0 to layer 6 against VGG_FACE.t7
        '''
        super(VGGFace, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))

        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

    def forward(self, x, mask=None):
        xx = x * 255.
        for j in range(3):
            xx[:, j, :, :] = xx[:, j, :, :] - meanvgg[j]

        if mask is not None:
            xx = xx * mask

        feature = {}
        feature['c1'] = self.conv1(xx)
        feature['r1'] = F.relu(feature['c1'])
        feature['c2'] = self.conv2(feature['r1'])
        feature['r2'] = F.relu(feature['c2'])
        feature['m2'] = self.maxpool(feature['r2'])
        feature['c3'] = self.conv3(feature['m2'])
        feature['r3'] = F.relu(feature['c3'])
        feature['c4'] = self.conv4(feature['r3'])
        feature['r4'] = F.relu(feature['c4'])
        feature['m4'] = self.maxpool(feature['r4'])
        feature['c5'] = self.conv5(feature['m4'])
        feature['r5'] = F.relu(feature['c5'])
        feature['c6'] = self.conv6(feature['r5'])
        feature['r6'] = F.relu(feature['c6'])
        feature['c7'] = self.conv7(feature['r6'])
        feature['r7'] = F.relu(feature['c7'])
        feature['m7'] = self.maxpool(feature['r7'])
        feature['c8'] = self.conv8(feature['m7'])
        feature['r8'] = F.relu(feature['c8'])
        feature['c9'] = self.conv9(feature['r8'])
        feature['r9'] = F.relu(feature['c9'])
        feature['c10'] = self.conv10(feature['r9'])
        feature['r10'] = F.relu(feature['c10'])
        return feature

    def forward_feat(self, x, mask=None):
        return self.forward(x, mask)

    def load_model(self, model_path):
        # Load pretrained VGG Face model
        vgg_dict = self.state_dict()
        pretrained_dict = torch.load(model_path)
        keys = vgg_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(keys, pretrained_keys):
            vgg_dict[k] = pretrained_dict[pk]

        self.load_state_dict(vgg_dict)


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
        self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
        self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
        self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
        self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())

    def load_model(self, model_file):
        vgg19_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)
        vgg19_keys = vgg19_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg19_keys, pretrained_keys):
            vgg19_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg19_dict)

    def forward(self, input_images, rgb2gray=False):
        '''
        :param images:
        :param rgb2gray: Set true to convert RGB to Gray
        :return:
        '''
        images = input_images.clone()
        if rgb2gray:
            images[:, 0, :, :] = input_images[:, 0, :, :] * 0.299 + input_images[:, 1, :, :] * 0.587 + input_images[:, 2, :, :] * 0.114
            images[:, 1, :, :] = images[:, 0, :, :]
            images[:, 2, :, :] = images[:, 0, :, :]

        for j in range(len(mean)):
            images[:, j, :, :] = (images[:, j, :, :] - mean[j]) / std[j]

        feature = {}
        feature['conv1_1'] = self.conv1_1(images)
        feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
        feature['pool1'] = self.pool1(feature['conv1_2'])
        feature['conv2_1'] = self.conv2_1(feature['pool1'])
        feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
        feature['pool2'] = self.pool2(feature['conv2_2'])
        feature['conv3_1'] = self.conv3_1(feature['pool2'])
        feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
        feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
        feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
        feature['pool3'] = self.pool3(feature['conv3_4'])
        feature['conv4_1'] = self.conv4_1(feature['pool3'])
        feature['conv4_2'] = self.conv4_2(feature['conv4_1'])

        return feature
