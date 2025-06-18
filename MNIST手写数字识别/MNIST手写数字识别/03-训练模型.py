# 超参数
import torch.optim

EPOCH = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 10

# 加载数据集
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# 创建张量转换器对象
to_tensor = ToTensor()
# 加载训练集，同时转换为张量格式
train_set = torchvision.datasets.MNIST("", transform=to_tensor, train=True, download=False)
# 转换为PyTorch训练的特定数据集格式
# 这里每10张图片作为一个批次，每次打乱
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# iterator = iter(train_loader)
# images, labels = next(iterator)
# print(images, labels)

# 创建模型、优化器、损失函数
# from FNN_MODEL import MyNet
from CNN_MODEL import MyNet

# 创建模型
model = MyNet()
# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 损失函数
loss_fn = torch.nn.NLLLoss()

# 训练训练
for epoch in range(EPOCH):
    print(f"当前纪元：{epoch + 1}/{EPOCH}")
    # 开启训练
    model.train(True)
    # 遍历一个批次的10张图片
    for i, (images, labels) in enumerate(train_loader):
        # 调整权重
        # 梯度清零
        optimizer.zero_grad(),
        # 前向传播计算损失
        outputs = model.forward(images)
        # 通过损失函数计算损失
        loss = loss_fn(outputs, labels)
        # 反向传播调节参数
        loss.backward()
        # 更新参数
        optimizer.step()
        print(f"当前批次：{i + 1}/{len(train_loader)}，当前损失值：{loss}")

# 保存模型参数
# torch.save(model.state_dict(), "FNN_MODEL_PARARM.pt")
torch.save(model.state_dict(), "CNN_MODEL_PARAM.pt")
