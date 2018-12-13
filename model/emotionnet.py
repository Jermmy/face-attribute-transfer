import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):

    def __init__(self, feat_size=17, pooling='max', dropout=0.5, feat_activation='linear'):
        '''
        :param feat_size: feature size of network
        :param pooling: pooling type
        :param dropout:
        :param feat_activation: activation of last layer
        '''
        super(EmotionNet, self).__init__()

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

        self.fc6 = nn.Sequential(nn.Linear(5 * 5 * 512, 1024),
                                 nn.Dropout(dropout))
        self.fc7 = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.Dropout(dropout))
        self.fc8 = nn.Linear(1024, feat_size)
        if feat_activation == 'linear':
            self.activation = None
        elif feat_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('Activation type [{:s}] is not supported'.format(feat_activation))

        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('Pooling type [{:s}] is not supported'.format(pooling))

    def forward(self, x):
        # 160 * 160 * 3
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pooling(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pooling(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.pooling(x)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.pooling(x)

        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.pooling(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        if self.activation:
            x = self.activation(x)

        return x

    def forward_feat(self, x):
        features = dict()
        features['c11'] = self.conv1_1(x)
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
        features['p43'] = self.pooling(features['r43'])

        features['c51'] = self.conv5_1(features['r43'])
        features['r51'] = F.relu(features['c51'])
        features['c52'] = self.conv5_2(features['r51'])
        features['r52'] = F.relu(features['c52'])
        features['c53'] = self.conv5_3(features['r52'])
        features['r53'] = F.relu(features['c53'])
        features['p53'] = self.pooling(features['r53'])

        features['r53'] = features['r53'].view(-1, 5 * 5 * 512)
        features['fc6'] = self.fc6(features['r53'] )
        features['r6'] = F.relu(features['fc6'])
        features['fc7'] = self.fc7(features['r6'])
        features['r7'] = F.relu(features['fc7'])
        features['fc8'] = self.fc8(features['r7'])

        if self.activation:
            features['r8'] = self.activation(features['fc8'])

        return features


class GLEmotionnet(nn.Module):

    def __init__(self, feat_size=17, pooling='max', dropout=0.5, feat_activation='linear'):
        super(GLEmotionnet, self).__init__()

        self.landmarkLayer = nn.Sequential(
            # 160 * 160 * 3
            nn.Conv2d(3, 80, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            # 160 * 160 * 80
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 80 * 80 * 80
            nn.Conv2d(80, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # 80 * 80 * 96
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 40 * 40 * 96
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 40 * 40 * 128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 40 * 40 * 128
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 20 * 20 * 128
        )
        self.landmarkFeat = nn.Sequential(
            nn.Linear(20 * 20 * 128, 1800),
            nn.Linear(1800, 1000),
            nn.Linear(1000, 68 * 2)
        )

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

        self.fc6 = nn.Sequential(nn.Linear(5 * 5 * 512 + 68 * 2, 1024),
                                 nn.Dropout(dropout))
        self.fc7 = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.Dropout(dropout))
        self.fc8 = nn.Linear(1024, feat_size)
        if feat_activation == 'linear':
            self.activation = None
        elif feat_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('Activation type [{:s}] is not supported'.format(feat_activation))

        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('Pooling type [{:s}] is not supported'.format(pooling))

    def forward(self, x):
        landmark_feat = self.landmarkLayer(x)
        landmark_feat = landmark_feat.view(landmark_feat.shape[0], -1)
        landmark_feat = self.landmarkFeat(landmark_feat)

        # 160 * 160 * 3
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pooling(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pooling(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pooling(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pooling(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pooling(x)

        x = x.view(x.shape[0], -1)
        # concat landmark_feat and au_feat
        x = torch.cat([landmark_feat, x], dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        if self.activation:
            x = self.activation(x)

        return x, landmark_feat

    def forward_feat(self, x):
        landmark_feat = self.landmarkLayer(x)
        landmark_feat = landmark_feat.view(landmark_feat.shape[0], -1)
        landmark_feat = self.landmarkFeat(landmark_feat)

        features = dict()
        features['c11'] = self.conv1_1(x)
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
        features['p43'] = self.pooling(features['r43'])

        features['c51'] = self.conv5_1(features['p43'])
        features['r51'] = F.relu(features['c51'])
        features['c52'] = self.conv5_2(features['r51'])
        features['r52'] = F.relu(features['c52'])
        features['c53'] = self.conv5_3(features['r52'])
        features['r53'] = F.relu(features['c53'])
        features['p53'] = self.pooling(features['r53'])

        features['p53'] = features['p53'].view(features['p53'].shape[0], -1)
        features['cat'] = torch.cat([landmark_feat, features['p53']], dim=1)
        features['fc6'] = self.fc6(features['cat'])
        features['r6'] = F.relu(features['fc6'])
        features['fc7'] = self.fc7(features['r6'])
        features['r7'] = F.relu(features['fc7'])
        features['fc8'] = self.fc8(features['r7'])

        if self.activation:
            features['r8'] = self.activation(features['fc8'])

        return features
