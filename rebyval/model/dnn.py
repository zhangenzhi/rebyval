import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):

        w_init = tf.random_normal_initializer(seed=100000)
        b_init = tf.zeros_initializer()

        self.w = tf.Variable(
            initial_value=w_init(
                shape=(input_shape[-1], self.units), dtype="float32"),
            trainable=True, name="w"
        )
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=True,
            name="b"
        )
            
    def build_fuse_layer(self, fuse_nums):
        
        w_init = tf.random_normal_initializer(seed=100000)
        b_init = tf.zeros_initializer()

        init_w = tf.stack(
            [w_init(shape=(self.w.shape[0], self.units), dtype="float32")] * fuse_nums)
        init_b = tf.stack([b_init(
            shape=(1, self.units), dtype="float32")] * fuse_nums)

        self.fuse_w = tf.Variable(
            initial_value=init_w, trainable=True, name="w")
        self.fuse_b = tf.Variable(
            initial_value=init_b, trainable=True,
            name="b"
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                ):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = self._build_fc()
        self.fc_act = self._build_act()
        self.fc_bn = self._build_bn()

    def _build_fc(self):
        layers = []
        for units in self.units:
            layers.append(Linear(units=units))
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
    
    def build_fuse_model(self, fuse_nums):
        pass

    def call(self, inputs):
        x = inputs
        x = self.flatten(x)
        for layer, act, bn in zip(self.fc_layers, self.fc_act, self.fc_bn):
            x = layer(x)
            x = act(x)
            # x = bn(x)
        return x