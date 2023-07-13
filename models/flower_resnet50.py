import tensorflow as tf

#Resnet50预训练模型微调
def Resnet50_model(input_shape, num_classes):
    ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)
    ResNet50.trainable = False
    
    model = tf.keras.models.Sequential()
    model.add(ResNet50)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model


