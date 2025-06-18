import tkinter as tk
from  tkinter import Canvas , Label

# import matplotlib.pyplot as plt
from  torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageOps

import numpy as np
import torch
# 引入模型
from CNN_MODEL import MyNet
# 定义类
class HandwritingApp:
    def __init__(self,root):
        self.root = root
        self.root.title("MNIST数字识别")
        # 创建Canvas
        self.canvas = Canvas(root,width=256,height=256,bg="white")
        self.canvas.pack(padx=10,pady=10)
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>" , self.on_draw_start) # 按下鼠标
        self.canvas.bind("<B1-Motion>",self.on_draw_move) # 移动鼠标

        # self.canvas.bind("<ButtonRelease-1>", self.on_button_release)# 松开鼠标

        # 创建一个空的PIL图像用于保存绘制结果
        self.drawing = Image.new("RGB",(256,256),'white')
        self.draw = ImageDraw.Draw(self.drawing)
        self.img_tensor = None
        # 识别按钮
        self.save_button = tk.Button(root , text="识别" ,
            command=self.on_save_button_clicked)
        self.save_button.pack(pady=20)
        # 清除按钮
        self.clear_button = tk.Button(root,text="清除",
            command=self.on_clear_button_clicked)
        self.clear_button.pack(pady=10)
        # 显示预测结果的标签
        self.prediction_label = Label(root,text="",width=20)
        self.prediction_label.pack(pady=20)
        # 加载模型参数
        self.model = MyNet()
        state_dict = torch.load("CNN_MODEL_PARAM.pt")
        self.model.load_state_dict(state_dict)

    def on_draw_start(self , event):
        # print("begin")
        self.lastx , self.lasty = event.x , event.y

    def on_draw_move(self , event):
        # print("move")
        x,y = event.x , event.y
        self.draw.line((self.lastx , self.lasty , x,y),
        fill='black' , width =10)
        self.canvas.create_line(self.lastx,
                              self.lasty , x,y,
            fill="black",width=10,capstyle=tk.ROUND,
            smooth=tk.TRUE, splinesteps=36)
        self.lastx , self.lasty = x,y

    # def on_button_release(self,event):
    #     print("end....")
    #     img_gray = self.drawing.convert("L")
    #     img_gray = ImageOps.invert(img_gray)
    #     resized_image = img_gray.resize((28, 28))
    #     resized_image.save("bb2.png")

    def save_as_tensor(self):
        # self.drawing.show()
        img_gray = self.drawing.convert("L")
        img_gray = ImageOps.invert(img_gray)
        resized_image = img_gray.resize((28,28))
        # resized_image.save("aa3.png")
        to_tensor = ToTensor()
        img_tensor = to_tensor(resized_image).float()
        self.img_tensor = img_tensor.unsqueeze(0)

    def on_save_button_clicked(self):
        self.save_as_tensor()
        outputs = self.model.forward(self.img_tensor)
        prediction = torch.argmax(outputs)
        self.prediction_label.config(text=f"识别数字:{int(prediction)}")
    def on_clear_button_clicked(self):
        self.canvas.delete("all")
        self.drawing = Image.new("RGB",(256,256),
                                 "white")
        self.draw = ImageDraw.Draw(self.drawing)
        self.img_tensor = None
        self.prediction_label.config(text="")

root = tk.Tk()
app = HandwritingApp(root)
root.mainloop()



