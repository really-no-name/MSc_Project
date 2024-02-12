from torch import nn
import torch.nn.functional as F


class UncertaintyCNN(nn.Module):
    def __init__(self):
        super(UncertaintyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 灰度图单通道输入 卷积层
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)  # 池化层，减少数据量



    def forward(self, x, apply_dropout=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        return label1_pred, label2_pred