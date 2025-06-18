import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
# from FNN_MODEL import MyNet
from CNN_MODEL import MyNet
img_path_list = [
    "真实图片/0.png",
    "真实图片/1.png",
    "真实图片/2.png",
    "真实图片/3.png",
    "真实图片/4.png",
    "真实图片/5.png",
    "真实图片/6.png",
    "真实图片/7.png",
    "真实图片/8.png",
    "真实图片/9.png",
]

# 创建张量转换器对象
to_tensor = ToTensor()
# 创建用于最终验证的张量集合对象
img_tensor_list = []
# 遍历图片，转换格式
for img_path in img_path_list:
    img = Image.open(img_path)
    # 修改为单通道灰度图
    img = img.convert("L")
    # 调整尺寸
    img = img.resize((28, 28))
    # 反转颜色
    img = ImageOps.invert(img)
    # 转换为张量
    img_tensor = to_tensor(img)
    # 添加到张量集合对象中
    img_tensor_list.append(img_tensor)

# 将张量集合对象转换为用于验证模型的张量对象
img_tensor_list = torch.stack(img_tensor_list)

# 创建模型，加载训练好的模型参数
model = MyNet()
# model.load_state_dict(torch.load("FNN_MODEL_PARARM.pt"))
model.load_state_dict(torch.load("CNN_MODEL_PARAM.pt"))
# 使用模型验证
outputs = model.forward(img_tensor_list)
prediction = torch.argmax(outputs, dim=1)
print(prediction)
