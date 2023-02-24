import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from cnn import CNN

# 批训练加速模型的训练速度
EPOCH = 1  # 定义批次训练的batch数
BATCH_SIZE = 50  # 定义批次训练的batch大小

LR = 0.001  # 学习率，学习率越高模型训练速度越快，但对应得会损失精度

cnn = CNN()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # 项目内容本质上是分类问题，针对多分类问题一般用交叉熵，因为它计算的是一个概率

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)  # 数据还是0-255之间
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.  # 该操作把255数据压缩到0-1之间
test_y = test_data.targets[:2000]
x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[
    :2000] / 255.  # unsqueeze 是加上了batchsize的维度，/255是压缩到0-1
y = test_data.targets[:2000]  # 为了节约时间只取前2000个

DOWNLOAD_MNIST = False
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # 是true下载训练集，如果是false 则下载测试集
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    # 载下来的数据转化为tensor格式（0-1），原始数据是pxl（0-255）
    download=DOWNLOAD_MNIST,
)
print(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)  # 需要把X，Y使用Variable转化为pytorch能处理的张量形式
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].squeeze(), 'real number')