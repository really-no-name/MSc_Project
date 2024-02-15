import torch
from torch import nn, optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from MyData import MyData
from UncertaintyCNN import UncertaintyCNN

transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms. ToTensor(),])

# 数据集和数据加载器
root_dir = 'all-mias/image'
csv_file = 'all-mias/info/info_clean.csv'
# dataset = MyData(root_dir, csv_file, transform=transform)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
train_set = MyData(root_dir, csv_file, mode='Train', transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=False)

test_set = MyData(root_dir, csv_file, mode='Test', transform=transform)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# Model, loss, Optimmizer
num_classes_label1 = 10
num_classes_label2 = 10
model = UncertaintyCNN(num_classes_label1, num_classes_label2)
criterion_label1 = nn.CrossEntropyLoss()
criterion_label2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练
# writer = SummaryWriter("log")
num_epochs = 10
for epoch in range(num_epochs):
    # step = 0
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        label1_preds, label2_preds = model(images)
        loss_label1 = criterion_label1(label1_preds, labels[:, 0])
        loss_label2 = criterion_label2(label2_preds, labels[:, 1])
        loss = loss_label1 + loss_label2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1} / {len(train_set)}], Loss: {loss.item():.4f}')
        # writer.add_images("Epoch:{}".format(epoch), images, step)
        # step = step + 1

# writer.close()