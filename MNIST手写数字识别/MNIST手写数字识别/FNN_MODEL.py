# 创建神经网络
# 创建一个类，继承torch.nn.Module
# 实现其中具体的隐藏层的内容
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 创建神经网络序列来定义隐藏层中的每一层
        self.sequential = nn.Sequential(
            # 1.张量展平
            # 将784个像素值保存到一个一位张量中
            # 原图张量形状为(1,28,28),经过展平后形状为(1,784)
            nn.Flatten(),
            # 2.经过全连接层，将784个输入转换为100个输出(总结为100个新特征)
            nn.Linear(784, 100),
            # 3.激活函数
            nn.ReLU(),
            # 4.经过全连接层，将100个输入转换为10个输出
            nn.Linear(100, 10),
            # 5.LogSoftmax函数，可以将10个输出映射到(0,1)的范围内，保证10个输出的结果之和为1
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)
