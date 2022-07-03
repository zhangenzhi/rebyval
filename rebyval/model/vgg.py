from re import X
import tensorflow as tf
from tensorflow.keras import layers

class VGG16(tf.keras.Model):
    
    def __init__(self, 
                 include_top=True,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax',
                 **kwargs):
        super(VGG16, self).__init__()
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation
        
        self._build()
        
    def _build(self):
        
        self.block_1 = []
        self.block_1.append(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1'))
        self.block_1.append(layers.BatchNormalization())
        self.block_1.append(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2'))
        self.block_1.append(layers.BatchNormalization())
        self.block_1.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        
        self.block_2 = []
        self.block_2.append(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1'))
        self.block_2.append(layers.BatchNormalization())
        self.block_2.append(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2'))
        self.block_2.append(layers.BatchNormalization())
        self.block_2.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
            
        self.block_3 = []
        self.block_3.append(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1'))
        self.block_3.append(layers.BatchNormalization())
        self.block_3.append(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2'))
        self.block_3.append(layers.BatchNormalization())
        self.block_3.append(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv3'))
        self.block_3.append(layers.BatchNormalization())
        self.block_3.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        
        self.block_4 = []
        self.block_4.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1'))
        self.block_4.append(layers.BatchNormalization())
        self.block_4.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2'))
        self.block_4.append(layers.BatchNormalization())
        self.block_4.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv3'))
        self.block_4.append(layers.BatchNormalization())
        self.block_4.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        
        self.block_5 = []
        self.block_5.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block5_conv1'))
        self.block_5.append(layers.BatchNormalization())
        self.block_5.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block5_conv2'))
        self.block_5.append(layers.BatchNormalization())
        self.block_5.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block5_conv3'))
        self.block_5.append(layers.BatchNormalization())
        self.block_5.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        
        self.top = []
        self.top.append(layers.Flatten(name='flatten'))
        self.top.append(layers.Dense(4096, activation='relu', kernel_initializer='he_normal', name='fc1'))
        self.top.append(layers.Dense(4096, activation='relu', kernel_initializer='he_normal', name='fc2'))
        self.top.append(layers.BatchNormalization())
        self.top.append(layers.Dense(self.classes, activation=self.classifier_activation, kernel_initializer='he_normal', name='predictions'))
    
    def call(self, input):
        
        x = input
        
        for layer in self.block_1:
            x = layer(x)
        
        for layer in self.block_2:
            x = layer(x)
        
        for layer in self.block_3:
            x = layer(x)
        
        for layer in self.block_4:
            x = layer(x)
            
        for layer in self.block_5:
            x = layer(x)
            
        for layer in self.top:
            x = layer(x)
            
        return x

class VGG11(tf.keras.Model):
    
    def __init__(self, 
                 include_top=True,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax',
                 **kwargs):
        super(VGG11, self).__init__()
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation
        
        self._build()
        
    def _build(self):
        
        self.block_1 = []
        self.block_1.append(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1'))
        self.block_1.append(layers.BatchNormalization())
        self.block_1.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        
        self.block_2 = []
        self.block_2.append(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1'))
        self.block_2.append(layers.BatchNormalization())
        self.block_2.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
            
        self.block_3 = []
        self.block_3.append(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1'))
        self.block_3.append(layers.BatchNormalization())
        self.block_3.append(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2'))
        self.block_3.append(layers.BatchNormalization())
        self.block_3.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        
        self.block_4 = []
        self.block_4.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1'))
        self.block_4.append(layers.BatchNormalization())
        self.block_4.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2'))
        self.block_4.append(layers.BatchNormalization())
        self.block_4.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        
        self.block_5 = []
        self.block_5.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block5_conv1'))
        self.block_5.append(layers.BatchNormalization())
        self.block_5.append(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='block5_conv2'))
        self.block_5.append(layers.BatchNormalization())
        self.block_5.append(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        
        self.top = []
        self.top.append(layers.Flatten(name='flatten'))
        self.top.append(layers.Dense(self.classes, activation=self.classifier_activation, kernel_initializer='he_normal', name='predictions'))
    
    def call(self, input):
        
        x = input
        
        for layer in self.block_1:
            x = layer(x)
        
        for layer in self.block_2:
            x = layer(x)
        
        for layer in self.block_3:
            x = layer(x)
        
        for layer in self.block_4:
            x = layer(x)
            
        for layer in self.block_5:
            x = layer(x)
            
        for layer in self.top:
            x = layer(x)
            
        return x
