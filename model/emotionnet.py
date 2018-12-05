import torch.nn as nn

class EmotionNet(nn.Module):

    def __init__(self):
        super(EmotionNet, self).__init__()

        self.conv1 = nn.Conv2d(64, 3, kernel_size=)