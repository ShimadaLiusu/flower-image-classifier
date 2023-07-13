import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class AlexNet:
    def __init__(self, input_shape=(227, 227, 3), num_classes=1000):
        self.model = self.build_model(input_shape, num_classes)
    
    def build_model(self, input_shape, num_classes):
        model = Sequential()
        
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        model.add(Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        model.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        
        model.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        model.add(Flatten())
        
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        return model

