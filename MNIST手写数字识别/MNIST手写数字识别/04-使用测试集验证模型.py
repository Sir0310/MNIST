#
import torch, torchvision
import matplotlib.pyplot as plt
from FNN_MODEL import MyNet
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
# 张量转换器
to_tensor = ToTensor()
# 加载测试集
test_set = torchvision.datasets.MNIST("", transform=to_tensor, train=False, download=False)
test_loader = DataLoader(test_set,shuffle=True)
# 迭代器
iterator = iter(test_loader)
# 加载模型
model = MyNet()
model.load_state_dict(torch.load("FNN_MODEL_PARARM.pt"))
# 循环9张图片
for i in range(9):
    images, labels = next(iterator)
    img = images[0]
    # 使用模型验证
    outputs = model.forward(img)
    # 取最大值，即认为的数字
    prediction = torch.argmax(outputs, dim=1)
    # 输出到九宫格
    plt.subplot(3, 3, i + 1)
    plt.title(int(prediction))
    plt.imshow(img[0],"gray")

plt.show()
