from torch import nn


class UncertaintyCNN(nn.Module):
    def __init__(self):
        super(UncertaintyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x, apply_dropout=False):
        return label1_pred, label2_pred