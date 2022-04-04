from random import seed
import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self, units=32, seed=100000, initial_value=None):
        super(Linear, self).__init__()
        self.units = units
        self.seed = seed
        self.initial_value = initial_value

    def build(self, input_shape):

        if self.initial_value == None:
            w_init = tf.random_normal_initializer(seed=self.seed)(shape=(input_shape[-1], self.units), dtype="float32")
            b_init = tf.zeros_initializer()(shape=(self.units,), dtype="float32")
        else:
            w_init = tf.cast(self.initial_value[0],dtype="float32")
            w_init = tf.reshape(w_init,shape=(input_shape[-1]))
            
            b_init = tf.cast(self.initial_value[1], dtype="float32")
            b_init = tf.reshape(b_init, shape=(self.units,))

        self.w = tf.Variable(
            initial_value=w_init,
            trainable=True, name="w"
        )
        self.b = tf.Variable(
            initial_value=b_init, trainable=True,
            name="b"
        )
        
    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                 use_bn=False,
                 seed=100000,
                 initial_value=None
                ):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.use_bn = use_bn 
        self.seed = seed
        self.initial_value = initial_value
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = self._build_fc()
        self.fc_act = self._build_act()
        self.fc_bn = self._build_bn()
        self.fc_bn = []

    def _build_fc(self):
        layers = []
        if self.initial_value != None:
            for i in range(len(self.units)):
                layers.append(Linear(units=self.units[i], seed=self.seed, initial_value=[i*2,i*2+1]))
        else:
            for units in self.units:
                layers.append(Linear(units=units, seed=self.seed))
        return layers
    
    def _build_bn(self):
        bn = []
        for _ in range(len(self.activations)-1):
            bn.append(tf.keras.layers.BatchNormalization())
        bn.append(tf.keras.layers.Lambda(lambda x:x))
        return bn

    def _build_act(self):
        acts = []
        for act in self.activations:
            acts.append(tf.keras.layers.Activation(act))
        return acts

    def call(self, inputs):
        x = inputs
        x = self.flatten(x)
        if self.use_bn:
            for layer, act, bn in zip(self.fc_layers, self.fc_act, self.fc_bn):
                x = layer(x)
                x = act(x)
                x = bn(x)
        else:
            for layer, act in zip(self.fc_layers, self.fc_act):
                x = layer(x)
                x = act(x)
        return x