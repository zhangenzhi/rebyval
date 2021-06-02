import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class ResNet(Model):
    def __init__(self, use_bias=False, pooling=None, classes=1000):
        super(ResNet, self).__init__()

        self.use_bias = use_bias
        self.pooling = pooling
        self.classes = classes

    def stack_fn(self, x):
        raise NotImplementedError

    def stack1(self, x, filters, blocks, stride1=2, name=None):
        x = self.block1(x, filters, stride=stride1, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = self.block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
        return x

    def block1(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):

        bn_axis = 3

        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)

        x = layers.Conv2D(
            filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)

        x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)

        return x

    def call(self, inputs):

        x = inputs

        x = layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=self.use_bias, name='conv1_conv')(x)

        x = self.stack_fn(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        import pdb
        pdb.set_trace()
        x = layers.Dense(self.classes, activations='softmax', name='prediction')(x)

        return x


class ResNet50(ResNet):
    def __init__(self, use_bias=False, pooling=None, classes=1000):
        super(ResNet50, self).__init__(use_bias, pooling, classes)

    def stack_fn(self, x):
        x = self.stack1(x, 64, 3, stride1=1, name='conv2')
        x = self.stack1(x, 128, 4, name='conv3')
        x = self.stack1(x, 256, 6, name='conv4')
        return self.stack1(x, 512, 3, name='conv5')
