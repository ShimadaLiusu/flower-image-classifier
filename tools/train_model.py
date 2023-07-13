import tensorflow as tf
import tensorflow.keras.datasets as datasets
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import sys
sys.path.append('.')
from models.flower_simple import FlowerModel
from models.flower_resnet50 import Resnet50_model
from models.flower_alexnet import AlexNet

# 可视化训练过程
def visualize(history):
    val_loss=history.history["val_loss"]
    train_loss=history.history["loss"]
    val_accuracy=history.history["val_accuracy"]
    train_accuracy=history.history["accuracy"]

    plt.figure()
    plt.plot(range(len(history.history["val_loss"])), history.history["val_loss"], label="val_loss", color="red")
    plt.plot(range(len(history.history["loss"])), history.history["loss"], label="train_loss", color="blue")
    plt.plot(range(len(history.history["val_accuracy"])), history.history["val_accuracy"], label="val_accuracy", color="green")
    plt.plot(range(len(history.history["accuracy"])), history.history["accuracy"], label="train_accuracy", color="orange")
    plt.legend()

    #Linux没有gui，要用这个保存图形
    plt.savefig('training_results.png')  # 保存图形为文件
    print("Training results saved as 'training_results.png'.")
    #Win用这个
    #plt.show()     
if __name__ == '__main__':
    
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    train_dir = "./dataset/train"  # 训练集文件夹路径
    val_dir = "./dataset/val"  # 验证集文件夹路径
    train_dir = pathlib.Path(train_dir)
    # 从数据集中获取所有花卉类别名称
    class_names = np.array([item.name for item in train_dir.glob('*') if item.is_dir()]) 

    # batch size 一次训练选择的样本数，要考虑GPU性能等，不宜太大
    BATCH_SIZE = 32
    IMG_HEIGHT = 227
    IMG_WIDTH = 227
    EPOCHS = 25

    # 加载训练集
    train_datagen = image.ImageDataGenerator(rescale=1./255)    # 数据增强和预处理
    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
    )

    # 加载验证集
    val_datagen = image.ImageDataGenerator(rescale=1./255)
    val_dataset = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
    )

    # 训练模型
    '''
    #最简单的
    model = FlowerModel(class_names) 
    
    #resnet50
    input_shape = (224, 224, 3)
    num_classes = len(class_names)
    model = Resnet50_model(input_shape, num_classes)
    model.summary()     #打印模型信息
    '''
    
    #AlexNet，网络输入为227*227*3
    model = AlexNet()

    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

    # 保存模型权重
    model.save_weights('./checkpoint/flower_model_weight')
    visualize(history)