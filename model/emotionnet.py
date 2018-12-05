import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):

    def __init__(self, pooling='max'):
        super(EmotionNet, self).__init__()

        self.conv1_1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
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

        self.fc6 = nn.Linear(5 * 5 * 512, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 17)

        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('Pooling type [{:s}] is not supported'.format(pooling))

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pooling(x, kernel_size=2, stride=2)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pooling(x, kernel_size=2, stride=2)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.pooling(x, kernel_size=2, stride=2)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)

        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)

        x = x.view(-1, 5 * 5 * 512)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        return x

    def forward_feat(self, x):
        features = dict()
        features['c11'] = self.conv1_1(x)
        features['r11'] = F.relu(features['c11'])
        features['c12'] = self.conv1_2(features['r11'])
        features['r12'] = F.relu(features['c12'])
        features['p12'] = self.pooling(features['r12'], kernel_size=2, stride=2)

        features['c21'] = self.conv2_1(features['p12'])
        features['r21'] = F.relu(features['c21'])
        features['c22'] = self.conv2_2(features['r21'])
        features['r22'] = F.relu(features['c22'])
        features['p22'] = self.pooling(features['r22'], kernel_size=2, stride=2)

        features['c31'] = self.conv3_1(features['p22'])
        features['r31'] = F.relu(features['c31'])
        features['c32'] = self.conv3_2(features['r31'])
        features['r32'] = F.relu(features['c32'])
        features['c33'] = self.conv3_3(features['r32'])
        features['r33'] = F.relu(features['c33'])
        features['p33'] = self.pooling(features['r33'], kernel_size=2, stride=2)

        features['c41'] = self.conv4_1(features['p33'])
        features['r41'] = F.relu(features['c41'])
        features['c42'] = self.conv4_2(features['r41'])
        features['r42'] = F.relu(features['c42'])
        features['c43'] = self.conv4_3(features['r42'])
        features['r43'] = F.relu(features['c43'])

        features['c51'] = self.conv5_1(features['r43'])
        features['r51'] = F.relu(features['c51'])
        features['c52'] = self.conv5_2(features['r51'])
        features['r52'] = F.relu(features['c52'])
        features['c53'] = self.conv5_3(features['r52'])
        features['r53'] = F.relu(features['c53'])

        features['r53'] = features['r53'].view(-1, 5 * 5 * 512)
        features['fc6'] = self.fc6(features['r53'])
        features['r6'] = F.relu(features['fc6'])
        features['fc7'] = self.fc7(features['r6'])
        features['r7'] = F.relu(features['fc7'])
        features['fc8'] = self.fc8(features['r7'])

        return features
