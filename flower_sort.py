import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import pathlib
import sys

from models.flower_simple import FlowerModel
from models.flower_resnet50 import Resnet50_model

#获取所有花卉类别
train_dir = "./dataset/train"  # 训练集文件夹路径
train_dir = pathlib.Path(train_dir)
class_names = np.array([item.name for item in train_dir.glob('*') if item.is_dir()]) 

# 加载模型权重
input_shape = (224, 224, 3)
num_classes = len(class_names)
model = Resnet50_model(input_shape, num_classes)
model.load_weights('./checkpoint/resnet50/flower_model_weight')

# 创建GUI窗口
window = tk.Tk()
window.title("花卉图像分类")
window.geometry("400x400")

# 选择图片按钮的回调函数
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # 显示选择的图片
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        # 进行图像分类
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # 图像预处理，与训练时一致
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        # 显示分类结果
        result_label.configure(text="预测结果: " + predicted_class)

# 创建选择图片按钮
select_button = tk.Button(window, text="选择图片", command=select_image)
select_button.pack(pady=10)

# 创建图像显示标签
image_label = tk.Label(window)
image_label.pack()

# 创建显示结果的标签
result_label = tk.Label(window, text="预测结果: ")
result_label.pack(pady=10)

window.mainloop()
