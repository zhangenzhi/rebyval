import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


class DenseNeuralNetwork(Model):
    def __init__(self, deep_dims, activations=None):
        super(DenseNeuralNetwork, self).__init__()
        self.deep_dims = deep_dims
        self.activations = activations if activations else ['relu']*len(deep_dims)
        self.dnn_layer = self.build_deep_layers()

    def build_deep_layers(self):
        deep_layers = []
        num_layers = len(self.deep_dims)

        for i in range(num_layers):
            dense_layer = Dense(
                self.deep_dims[i],
                activation=self.activations[i],
                kernel_initializer=tf.keras.initializers.GlorotUniform())
            deep_layers.append(dense_layer)

        return deep_layers

    def call(self, x):
        next_input = Flatten()(x)
        # next_input = next_input * next_input * 5
        for deep_layer in self.dnn_layer:
            next_input = deep_layer(next_input)
        return next_input
