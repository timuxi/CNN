from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 先找到CNN的父类（比如是类A），然后把类CNN的对象self转换为类A的对象，然后“被转换”的类A对象调用类A对象自己的__init__函数.
        self.conv1 = nn.Sequential(
            # 卷积+激活+池化
            # 过滤器  高度-filter 用于提取出卷积出来的特征的属性
            # 图片的维度是 (1，28，28)1是chanel的维度，28x28的长宽高
            nn.Conv2d(
                1,  # in_channels=1 图片的通道数 灰度图为1，如果为rgb图像则为3
                16,  # out_channels=16 多少个输出的高度(filter的个数)
                5,  # kernel_size=5，filter的高宽都是5个像素点
                1,  # stride=1，卷积步长
                2,  # padding=2填充，如果想输出的和输入的长宽一样则需要padding=(kernel_size-1)/2
            ),  # (16,28,28)
            nn.ReLU(),
            # 删选重要信息，参数(),kernel_size=2,把2x2的区域中选最大的值变成1x1
            nn.MaxPool2d(kernel_size=2),  # (16,14,14)
        )
        # (16,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            # 还可以用AvgPool3d，但一般用最大
            nn.MaxPool2d(2),  # (32,7,7)
        )
        # 输出层
        self.out = nn.Linear(32 * 7 * 7, 10)  # （a，b）a是数据维度 b是分类器有十个

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # (batch,32*7*7)
        output = self.out(x)
        return output
