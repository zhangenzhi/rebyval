import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


class dnn(Model):
    def __init__(self, hidden_dims=[128, 64, 1]):
        super(dnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.dnn_layer = self.build_deep_layers()

    def build_deep_layers(self):
        deep_layers = []

        for dim in self.hidden_dims:
            dense_layer = Dense(
                dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.keras.initializers.GlorotUniform())
            deep_layers.append(dense_layer)

        return deep_layers

    def call(self, x):
        next_input = Flatten()(x)
        # next_input = next_input * next_input * 5
        for deep_layer in self.dnn_layer:
            next_input = deep_layer(next_input)
        return next_input
