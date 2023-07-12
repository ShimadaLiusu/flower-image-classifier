import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
import numpy as np
import pathlib

import sys
sys.path.append('.')
from models.flower_simple import FlowerModel
from models.flower_resnet50 import Resnet50_model

def load_test_data(data_path):
    test_dir = pathlib.Path(data_path)
    test_datagen = image.ImageDataGenerator(rescale=1./255)
    test_dataset = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    return test_dataset

def evaluate(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    return loss, accuracy

if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    test_dir = "./dataset/test"  
    test_dataset = load_test_data(test_dir)

    # 加载模型
    input_shape = (224, 224, 3)
    num_classes = len(test_dataset.class_indices)
    model = Resnet50_model(input_shape, num_classes)
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.load_weights('./checkpoint/flower_model_weight')

    # 模型评估
    loss, accuracy = evaluate(model, test_dataset)

    # 打印评估结果
    print("Evaluation Result:")
    print("Loss:", loss)
    print("Accuracy:", accuracy)
