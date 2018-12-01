import torch.nn as nn
import torch
import torch.nn.functional as F

class GramLoss(nn.Module):

    def __init__(self):
        super(GramLoss, self).__init__()
        self.gram_loss = nn.MSELoss()

    def _gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, x, y):
        gram_x = self._gram_matrix(x)
        gram_y = self._gram_matrix(y)
        loss = self.gram_loss(gram_x, gram_y)
        return loss