import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# 读取 CSV 文件
csv_file_path = 'all-mias/info/info_clean.csv'
df = pd.read_csv(csv_file_path)

# 将类别标签转换为数字
label_encoder = LabelEncoder()
df['Class_of_abnormality_present_encoded'] = label_encoder.fit_transform(df['Class_of_abnormality_present'].astype(str))

# 数据加载类
class MyData(Dataset):
    def __init__(self, root_dir, data_frame, transform=None):
        self.root_dir = root_dir
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx]['Class_of_abnormality_present_encoded']

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

# 设置转换
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建数据集
train_df = df[df['TrainTest'] == 'Train']
test_df = df[df['TrainTest'] == 'Test']

train_dataset = MyData(root_dir='all-mias/image', data_frame=train_df, transform=transform)
test_dataset = MyData(root_dir='all-mias/image', data_frame=test_df, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN 网络类
class UncertaintyCNN(torch.nn.Module):
    def __init__(self):
        super(UncertaintyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(32 * 256 * 256, 500)
        self.fc2 = torch.nn.Linear(500, len(label_encoder.classes_))  # 以类别数量作为输出层大小
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 256 * 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型和优化器
model = UncertaintyCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试过程
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'准确度: {100 * correct / total}%')
