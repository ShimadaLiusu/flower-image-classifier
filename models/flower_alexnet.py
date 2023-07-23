import tensorflow as tf
import tensorflow.keras.layers as layers

class AlexNet(tf.keras.models.Model):
    def __init__(self, class_names=[]):
        super(AlexNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")

        self.conv2 = layers.Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation="relu")
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")

        self.conv3 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu")

        self.conv4 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu")

        self.conv5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool5 = layers.MaxPool2D(pool_size=(3, 3),1 strides=(2, 2), padding="valid")

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation="relu")
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(4096, activation="relu")
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(len(class_names), activation="softmax")  # 输出层的层数是类别数

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
