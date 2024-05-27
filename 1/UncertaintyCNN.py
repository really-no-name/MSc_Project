# CNN神经网络

from torch import nn
import torch.nn.functional as F


class UncertaintyCNN(nn.Module):
    def __init__(self, num_classes_label1, num_calsses_label2):
        super(UncertaintyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 灰度图单通道输入 卷积层
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)  # 池化层，减少数据量
        self.dropout = nn.Dropout(p = 0.5)
        self.fc_1 = nn.Linear(64*128*128, 256)
        self.fc_label1 = nn.Linear(256, num_classes_label1)
        self.fc_label2 = nn.Linear(256, num_calsses_label2)


    def forward(self, x, apply_dropout=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 128 * 128)  # 这是 PyTorch 中用来重塑张量的方法。它会返回一个新的张量，其数据与原始张量相同，但形状不同。`-1`: 当在 `.view()` 中作为维数之一使用时，-1 意味着这个特定的维数将被自动计算。它的意思是 "根据其他维度和张量中元素的总数相应地调整这个维度"。`64 * 128 * 128`: 这些数字代表你想赋予张量的新形状。在本例中，张量 `x` 将被重塑为第二个维度为 `64 * 128 * 128` 的形状，等于 1,048,576。
        x = F.relu(self.fc_1(x))
        if apply_dropout:
            x = self.dropout(x)
        label1_pred = self.fc_label1(x)
        label2_pred = self.fc_label2(x)
        return label1_pred, label2_pred