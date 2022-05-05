import tensorflow as tf
from tensorflow import keras

class CNN(tf.keras.Model):
    def __init__(self,
                 kernels=[3, 3, 3, 3],
                 filters=[64, 32, 16, 8],
                 activations=['relu', 'relu', 'relu', 'relu'],
                 tail_acts='softmax',
                 classes=10
                 ):
        super(CNN, self).__init__()

        self.kernels = kernels
        self.filters = filters
        self.activations = activations

        self.flatten = keras.layers.Flatten()
        
        self.cnn_layers = self._build_cnn()
        self.cnn_act = self._build_act()
        self.cnn_bn = self._build_bn()
        
        self.mlp = keras.layers.Dense(units=classes)
        self.last_act = keras.layers.Activation(tail_acts)

    def _build_cnn(self):
        layers = []
        for kernel_size, filters in zip(self.kernels, self.filters):
            layers.append(keras.layers.Conv2D(
                filters=filters, kernel_size=kernel_size))
        return layers

    def _build_bn(self):
        bn = []
        for _ in range(len(self.activations)-1):
            bn.append(keras.layers.BatchNormalization())
        bn.append(keras.layers.Lambda(lambda x: x))
        return bn

    def _build_act(self):
        acts = []
        for act in self.activations:
            acts.append(keras.layers.Activation(act))
        return acts


    def call(self, inputs):
        x = inputs
        
        # cnn
        for cnn, act, bn in zip(self.cnn_layers, self.cnn_act, self.cnn_bn):
            x = cnn(x)
            x = act(x)
            x = bn(x)
            
        # flatten
        x = self.flatten(x)
        
        # mlp
        x = self.mlp(x)
        x = self.last_act(x)
        
        return x