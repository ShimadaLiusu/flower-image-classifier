import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import pathlib
import sys
import h5py

from models.flower_simple import FlowerModel
from models.flower_resnet50 import Resnet50_model
from models.flower_alexnet import AlexNet

#获取所有花卉类别
train_dir = "./dataset/train"
train_dir = pathlib.Path(train_dir)
class_names = np.array([item.name for item in train_dir.glob('*') if item.is_dir()]) 

# 加载模型权重
IMG_HEIGHT = 224
IMG_WIDTH = 224
input_shape = (224, 224, 3)
num_classes = len(class_names)
model = Resnet50_model(input_shape, num_classes)
model.load_weights('./checkpoint/resnet50/flower_model_weight')

# 创建GUI窗口
window = tk.Tk()
window.title("花卉类别检测系统")
window.geometry("330x450")

# 设置按钮样式
button_style = {"font": ("微软雅黑", 10), "padx": 10, "pady": 5}

# 选择图片
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # 显示图片
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        # 保存图片路径
        global selected_image_path
        selected_image_path = file_path

# 图像分类检测
def classify_image():
    if selected_image_path:     #用户已经选择图像后可以分类
       
        img = tf.keras.preprocessing.image.load_img(selected_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # 图像预处理，与训练时一致
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]    #预测结果

        class_mapping = {
            "daisy": "小雏菊",
            "iris": "鸢尾花",
            "rose": "玫瑰",
            "sunflower": "向日葵",
            "wild_rose": "蔷薇",
            "tulips": "郁金香"
        }

        # 获取英文名对应的中文
        predicted_class_cn = class_mapping.get(predicted_class, predicted_class)

        # 显示分类结果
        result_label.configure(text="分类结果: " + predicted_class_cn)
        

# 创建选择图片按钮
select_button = tk.Button(window, text="选择图片", command=select_image, **button_style)
select_button.grid(row=0, column=0, padx=10, pady=10)

# 创建分类按钮
classify_button = tk.Button(window, text="分类", command=classify_image, **button_style)
classify_button.grid(row=0, column=1, padx=10, pady=10)

# 创建显示图像的标签
image_label = tk.Label(window)
image_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# 创建显示结果的标签
result_label = tk.Label(window, text="分类结果: ", font=("微软雅黑", 14, "bold"))
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="s")

window.mainloop()