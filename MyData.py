import os  # 与操作系统进行交互
import pandas as pd  # 处理 CSV 文件
from PIL import Image  # (Python Imaging Library) 用于图像处理
from torch.utils.data import Dataset  # 是 PyTorch 中用于创建自定义数据集的基类


class MyData(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file)
        self.img_names = self.csv_file.iloc[:, 0]  # 假设第一列是图片名称
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('L')  # 确保灰度图

        if self.transform:
            img = self.transform(img)

        # 读取 label_1 到 label_6
        labels = self.csv_file.iloc[index, 1:7].values
        return img, labels

    def __len__(self):
        return len(self.img_names)


root_dir = 'all-mias/image'
csv_file = 'all-mias/info/info_clean.csv'
mini_mias = MyData(root_dir, csv_file)
img, label = mini_mias[0]
img.show()
print(label)
