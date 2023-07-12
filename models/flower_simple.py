import tensorflow as tf
import tensorflow.keras.layers as layers

class FlowerModel(tf.keras.models.Model):
    def __init__(self, class_names=[]):
        super(FlowerModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(64, activation="relu")
        self.fc3 = layers.Dense(len(class_names), activation="softmax")  # 输出层的层数是花卉类别数

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


