
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class ResNet(Model):
    def __init__(self, use_bias=True, pooling=None, include_top=True, classes=1000, preact=False,
                 regularizer=None, name='resnet', **kwargs):
        super(ResNet, self).__init__()

        self.use_bias = use_bias
        self.pooling = pooling
        self.include_top = include_top
        self.classes = classes
        self.preact = preact
        self.regularizer = regularizer

        self.preprocess_layers = self._build_preprocess()
        self.stack_fn_stacks = self._build_stack_fn()
        self.dense_inference_layers = self._build_dense_inference()

    def _build_stack_fn(self, name=None):
        raise NotImplementedError

    def stack_fn(self, x, stack1s):
        raise NotImplementedError

    def _build_preprocess(self):
        preprocess_layers = []
        preprocess_layers.append(layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad'))
        preprocess_layers.append(
            layers.Conv2D(64, 7, strides=2, kernel_initializer='he_normal', use_bias=self.use_bias,
                          kernel_regularizer=self.regularizer,
                          name='conv1_conv'))
        if not self.preact:
            preprocess_layers.append(layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn'))
            preprocess_layers.append(layers.Activation('relu', name='conv1_relu'))
        preprocess_layers.append(layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad'))
        preprocess_layers.append(layers.MaxPool2D(pool_size=3, strides=2, name='pool1_pool'))
        return preprocess_layers

    def _preprocess(self, x, process_layers):
        for pre_layer in process_layers:
            x = pre_layer(x)
        return x

    def _build_dense_inference(self):
        inference_layer = []

        if self.preact:
            inference_layer.append(layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='post_conv1_bn'))
            inference_layer.append(layers.Activation('relu', name='post_conv1_relu'))


        if self.include_top:
            inference_layer.append(layers.GlobalAveragePooling2D(name='avg_pool'))
            inference_layer.append(
                layers.Dense(self.classes, activation='softmax', name='prediction', kernel_initializer='he_normal',
                             kernel_regularizer=self.regularizer))
        else:
            inference_layer.append(layers.GlobalAveragePooling2D(name='avg_pool'))
        return inference_layer

    def _dense_inference(self, x, dense_inference_layers):
        for layer in dense_inference_layers:
            x = layer(x)
        return x

    def _build_stack1(self, filters, blocks, strides1=2, name=None):
        seq_layers_stack1 = []
        seq_layers_stack1.append(self._build_block1(filters, strides=strides1, name=name + '_block1'))
        for i in range(2, blocks + 1):
            seq_layers_stack1.append(
                self._build_block1(filters, conv_shortcut=False, name=name + '_block' + str(i)))
        return seq_layers_stack1

    def stack1(self, x, seq_layers_stack1):
        for block, shortcut in seq_layers_stack1:
            x = self.block1(x, block, shortcut)
        return x

    def _build_block1(self, filters, kernel_size=3, strides=1, conv_shortcut=True, zeropad_shortcut=False, name=None):
        seq_layers_block = []
        seq_layer_shortcut = []
        bn_axis = 3

        if conv_shortcut:
            seq_layer_shortcut.append(layers.Conv2D(
                4 * filters, 1, strides=strides, name=name + '_0_conv',
                kernel_regularizer=self.regularizer))
            seq_layer_shortcut.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn'))
        elif zeropad_shortcut:
            def depth_pad(x):
                new_channels = 4 * filters
                output = tf.identity(x)
                repetitions = new_channels / x.shape.as_list[-1]
                for _ in range(int(repetitions)):
                    zeroTensor = tf.zeros_like(x, name='pad_depth1')
                    output = tf.keras.backend.concatenate([output, zeroTensor])
                return output

            seq_layer_shortcut.append(layers.MaxPool2D(pool_size=(1, 1), strides=(strides, strides), padding='same'))
            seq_layer_shortcut.append(layers.Lambda(lambda x: depth_pad(x)))
        else:
            seq_layer_shortcut.append(layers.Lambda(lambda x: x))

        seq_layers_block.append(
            layers.Conv2D(filters, 1, name=name + '_1_conv', strides=1, kernel_initializer='he_normal',
                          kernel_regularizer=self.regularizer))
        seq_layers_block.append(layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn'))
        seq_layers_block.append(layers.Activation('relu', name=name + '_1_relu'))

        seq_layers_block.append(
            layers.Conv2D(filters, kernel_size, strides=strides, padding='SAME', name=name + '_2_conv',
                          kernel_initializer='he_normal',
                          kernel_regularizer=self.regularizer))
        seq_layers_block.append(layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn'))
        seq_layers_block.append(layers.Activation('relu', name=name + '_2_relu'))

        seq_layers_block.append(layers.Conv2D(4 * filters, 1, name=name + '_3_conv', kernel_initializer='he_normal',
                                              kernel_regularizer=self.regularizer))
        seq_layers_block.append(layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn'))

        seq_layers_block.append(layers.Add(name=name + '_add'))
        seq_layers_block.append(layers.Activation('relu', name=name + '_out'))

        return seq_layers_block, seq_layer_shortcut

    def block1(self, x, seq_layers_block, seq_layer_shortcut):

        shortcut = x
        for layer in seq_layer_shortcut:
            shortcut = layer(shortcut)

        for layer in seq_layers_block:
            x = layer(x) if layer.name[-3:] != 'add' else layer([shortcut, x])

        return x

    def call(self, inputs):

        x = inputs
        

        x = self._preprocess(x, self.preprocess_layers)

        x = self.stack_fn(x, self.stack_fn_stacks)

        x = self._dense_inference(x, self.dense_inference_layers)

        return x


class ResNet50(ResNet):
    def __init__(self, use_bias=True, pooling=None, classes=1000, 
                 regularizer=tf.keras.regularizers.l2(l2=0.0001), name='ResNet50',
                 **kwargs):
        super(ResNet50, self).__init__(use_bias=use_bias, pooling=pooling, classes=classes, regularizer=regularizer, name=name)

    def _build_stack_fn(self, name='resnet50'):
        seq_layer_stacks = []

        seq_layer_stacks.append(self._build_stack1(64, 3, strides1=1, name=name + '_conv2'))
        seq_layer_stacks.append(self._build_stack1(128, 4, name=name + '_conv3'))
        seq_layer_stacks.append(self._build_stack1(256, 6, name=name + '_conv4'))
        seq_layer_stacks.append(self._build_stack1(512, 3, name=name + '_conv5'))

        return seq_layer_stacks

    def stack_fn(self, x, stack_fn_stacks):
        for stack in stack_fn_stacks:
            x = self.stack1(x, stack)
        return x


class ResNetV2(Model):
    # For cifar10 training.
    def __init__(self, 
                 use_bias=True, 
                 pooling=None, 
                 include_top=True, 
                 classes=10, 
                 regularizer=None, 
                 name='resnetv2', 
                 **kwargs):
        super(ResNetV2, self).__init__()

        self.use_bias = use_bias
        self.pooling = pooling
        self.include_top = include_top
        self.classes = classes
        self.regularizer = regularizer

        self.preprocess_layers = self._build_preprocess()
        self.stack_fn_stacks = self._build_stack_fn()
        self.dense_inference_layers = self._build_dense_inference()
    
    def _build_stack_fn(self, name=None):
        raise NotImplementedError

    def stack_fn(self, x, stack1s):
        raise NotImplementedError
    
    def _build_preprocess(self):
        preprocess_layers = []
        preprocess_layers.append(
            layers.Conv2D(16, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', use_bias=self.use_bias,
                          kernel_regularizer=self.regularizer, name='pre_conv'))
        preprocess_layers.append(layers.BatchNormalization(name='pre_conv_bn'))
        preprocess_layers.append(layers.Activation('relu', name='pre_act_relu'))
        return preprocess_layers 
    
    def _preprocess(self, x, process_layers):
        for pre_layer in process_layers:
            x = pre_layer(x)
        return x
    
    def _build_dense_inference(self):
        inference_layer = []
        if self.include_top:
            inference_layer.append(layers.GlobalAveragePooling2D(name='avg_pool'))
            inference_layer.append(layers.Flatten())
            inference_layer.append(
                layers.Dense(self.classes, activation='softmax', name='prediction', kernel_initializer='he_normal',
                             kernel_regularizer=self.regularizer))
        else:
            inference_layer.append(layers.GlobalAveragePooling2D(name='avg_pool'))
        return inference_layer
    
    def _dense_inference(self, x, dense_inference_layers):
        for layer in dense_inference_layers:
            x = layer(x)
        return x
    
    def _build_stack1(self, filters, blocks, s_downsampling=2, name=None):
        seq_layers_stack1 = []
        seq_layers_stack1.append(self._build_block1(filters, kernel_size=1, strides=s_downsampling, name=name + '_block1'))
        for i in range(2, blocks + 1):
            seq_layers_stack1.append(
                self._build_block1(filters, kernel_size=3, strides=1, name=name + '_block' + str(i)))
        return seq_layers_stack1

    def stack1(self, x, seq_layers_stack1):
        for block, shortcut in seq_layers_stack1:
            x = self.block1(x, block, shortcut)
        return x
    
    def _build_block1(self, filters, kernel_size, strides, name=None):
        seq_layers_block = []
        seq_layer_shortcut = []
        bn_axis = 3

        if strides == 2:
            seq_layer_shortcut.append(layers.Conv2D(filters, kernel_size=(1,1), padding='same', strides=strides,
                                                    name=name + '_downsampling_skip', 
                                                    kernel_initializer='he_normal',
                                                    kernel_regularizer=self.regularizer))
            # seq_layer_shortcut.append(layers.BatchNormalization(name=name + '_downsampling_skip_bn'))
        else:
            seq_layer_shortcut.append(layers.Lambda(lambda x: x))

        seq_layers_block.append(
            layers.Conv2D(filters, kernel_size=(3,3), padding='same', name=name + '_1_conv', strides=strides, 
                          kernel_initializer='he_normal',
                          kernel_regularizer=self.regularizer))
        seq_layers_block.append(layers.BatchNormalization(name=name + '_1_bn'))
        seq_layers_block.append(layers.Activation('relu', name=name + '_1_relu'))

        seq_layers_block.append(
            layers.Conv2D(filters, kernel_size=(3,3), strides=1, padding='same', name=name + '_2_conv',
                          kernel_initializer='he_normal',
                          kernel_regularizer=self.regularizer))
        seq_layers_block.append(layers.BatchNormalization(name=name + '_2_bn'))

        seq_layers_block.append(layers.Add(name=name + '_add'))
        seq_layers_block.append(layers.Activation('relu', name=name + '_out'))

        return seq_layers_block, seq_layer_shortcut
    
    def block1(self, x, seq_layers_block, seq_layer_shortcut):

        shortcut = x
        for layer in seq_layer_shortcut:
            shortcut = layer(shortcut)

        for layer in seq_layers_block:
            x = layer(x) if layer.name[-3:] != 'add' else layer([shortcut, x])

        return x
    
    def call(self, inputs):

        x = inputs

        x = self._preprocess(x, self.preprocess_layers)

        x = self.stack_fn(x, self.stack_fn_stacks)

        x = self._dense_inference(x, self.dense_inference_layers)

        return x

class ResNet56(ResNetV2):
    def __init__(self, 
                 use_bias=True, 
                 pooling=None, 
                 classes=10, 
                 regularizer=tf.keras.regularizers.l2(1e-4), 
                 name='ResNet56',
                 **kwargs):
        super(ResNet56, self).__init__(use_bias=use_bias, 
                                       pooling=pooling, 
                                       classes=classes, 
                                       regularizer=regularizer, 
                                       name=name)

    def _build_stack_fn(self, name='ResNet56'):
        seq_layer_stacks = []
        
        seq_layer_stacks.append(self._build_stack1(16, 9, s_downsampling=1, name=name + '_stack1'))
        seq_layer_stacks.append(self._build_stack1(32, 9, name=name + '_stack2'))
        seq_layer_stacks.append(self._build_stack1(64, 9, name=name + '_stack3'))

        return seq_layer_stacks

    def stack_fn(self, x, stack_fn_stacks):
        for stack in stack_fn_stacks:
            x = self.stack1(x, stack)
        return x

def nResNet56(name, classes):
    from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten
    from tensorflow.keras import Model
    from tensorflow.keras.regularizers import l2

    n = 9 # 56 layers
    channels = [16, 32, 64]

    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(channels[0], kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    for c in channels:
        for i in range(n):
            subsampling = i == 0 and c > 16
            strides = (2, 2) if subsampling else (1, 1)
            y = Conv2D(c, kernel_size=(3, 3), padding="same", strides=strides, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)
            y = Activation(tf.nn.relu)(y)
            y = Conv2D(c, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)        
            if subsampling:
                x = Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
            x = Add()([x, y])
            x = Activation(tf.nn.relu)(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(10, activation=tf.nn.softmax, kernel_initializer="he_normal")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.type = "resnet" + str(6 * n + 2)
    return model
    