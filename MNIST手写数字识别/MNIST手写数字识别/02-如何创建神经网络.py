import torch.nn


# 创建一个类，继承torch.nn.Module，即可称为一个模型类的子类
class MyNet(torch.nn.Module):
    # 需要写出构造方法，在构造方法中调用父类的构造方法
    def __init__(self):
        super(MyNet, self).__init__()
        print("执行自定义的神经网络模型的构造方法，通常在这里定义神经网络中的隐藏层")

    # 前向传播，用于接收参数，得到输出。这里的x就是输入
    def forward(self, x):
        print("执行自定义神经网络模型的前向传播方法，用于操作输入的参数")
        return x ** 2


model = MyNet()
output = model.forward(5)
print(output)