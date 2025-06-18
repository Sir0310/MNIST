# 创建卷积神经网络
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.seq = nn.Sequential(
            # 设计卷积神经网络中的每一层
            # 第一次卷积，使用10个3*3卷积核，得到10个新特征
            # 28*28 -->  28-3+1  --> 26*26
            nn.Conv2d(1, 10, 3),
            # 激活函数
            nn.ReLU(),
            # 第一次池化，使用2*2窗口，每次移动2个步长
            nn.MaxPool2d(2, 2),
            # 26*26 --> 13*13
            # 第二次卷积，使用20个5*5卷积核，得到20个新特征
            nn.Conv2d(10, 20, 5),
            # 13*13 --> 13-5+1 -->9*9
            # 展平
            nn.Flatten(),
            # 第一个全连接20*9*9 --> 100
            nn.Linear(20 * 9 * 9, 100),
            # 激活函数
            nn.ReLU(),
            # 第二个全连接100 --> 10
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)
