from torch.utils.tensorboard import SummaryWriter
from MyData import MyData
from torch.utils.data import DataLoader

# transform =

# 数据集和数据加载器
root_dir = 'path_to_images'
csv_file = 'path_to_csv'
dataset = MyData(root_dir, csv_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# 训练
writer = SummaryWriter("log")
num_epochs = 10
for epoch in range(num_epochs):
    step = 0
    for i, (images, labels) in enumerate(data_loader):
        writer.add_images("Epoch:{}".format(epoch), images, step)
        step = step + 1

writer.close()