import torchvision

# torchvision.datasets.MNIST("下载路径",train=True表示训练集False表示测试集,download=True表示下载)
# 下载训练集
train_set = torchvision.datasets.MNIST("", train=True, download=True)
# 下载测试集
test_set = torchvision.datasets.MNIST("", train=False, download=True)

print(train_set)
print(test_set)

# 取训练集中的第一个样本，返回图片和标签
image, label = train_set[0]
import matplotlib.pyplot as plt
plt.imshow(image, cmap="gray")
plt.title(label)
plt.show()


# 使用张量的形式查看其中的一个样本
# 图像转换为张量
from torchvision.transforms import ToTensor
# 创建张量转换器对象
toTensor = ToTensor()
# 转换图片为张量
image_tensor = toTensor(image)
# 张量的形状(1,28,28)表示单通道28*28
print(image_tensor.shape)
# 张量中的数据
# 0~1表示亮度 1最亮 0最暗
print(image_tensor)


