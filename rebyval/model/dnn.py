from random import seed
import tensorflow as tf
from tensorflow import keras
from rebyval.train.utils import ForkedPdb

class Linear(keras.layers.Layer):
    def __init__(self, units=32, seed=100000, initial_value=None):
        
        super(Linear, self).__init__()
        self.seed=seed
        self.units = units
        self.initial_value = initial_value
        
        self.build_from_value()

    def build(self, input_shape):
        if self.initial_value == None:
            w_init = tf.random_normal_initializer(seed=self.seed)(shape=(input_shape[-1], self.units), dtype="float32")
            b_init = tf.zeros_initializer()(shape=(self.units,), dtype="float32")

            self.w = tf.Variable(
                initial_value=w_init,
                trainable=True, name="w",
                constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
            )
            self.b = tf.Variable(
                initial_value=b_init, trainable=True,
                name="b",
                constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
            )
        
    def build_from_value(self):
        if self.initial_value!=None:
            
            w_init = tf.cast(self.initial_value[0], dtype="float32")
            b_init = tf.cast(self.initial_value[1], dtype="float32")
            b_init = tf.reshape(b_init, shape=(self.units,))
            
            self.w = tf.Variable(
                initial_value=w_init,
                trainable=True, name="w",
                constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
            )
            self.b = tf.Variable(
                initial_value=b_init, trainable=True,
                name="b",
                constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
            )
        
    def call(self, inputs):
        # ForkedPdb().set_trace()
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                 use_bn=False,
                 embedding=False,
                 seed=100000,
                 initial_value=None,
                 **kwargs
                ):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.use_bn = use_bn
        self.embedding = embedding 
        self.seed = seed
        self.initial_value = initial_value
        
        if embedding:
            self._build_emb()
            
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = self._build_fc()
        self.fc_act = self._build_act()
        self.fc_bn = self._build_bn()
        self.fc_bn = []
    
    def _build_emb(self):
       
        if self.initial_value != None:
            w_s = self.initial_value.pop(0)
            b_s = self.initial_value.pop(0)
            w_a = self.initial_value.pop(0)
            b_a = self.initial_value.pop(0)
            w_t = self.initial_value.pop(0)
            b_t = self.initial_value.pop(0)
            self.state_emb = Linear(units=256, 
                                     seed=self.seed, 
                                     initial_value=[w_s, b_s])
            self.action_emb   = Linear(units=1024, 
                                     seed=self.seed, 
                                     initial_value=[w_a, b_a])
            self.step_emb   = Linear(units=256, 
                                     seed=self.seed, 
                                     initial_value=[w_t, b_t])
            
        else:
            self.state_emb = Linear(units=256)
            self.action_emb = Linear(units=1024)
            self.step_emb   = Linear(units=256)

        
    def _build_fc(self):
        layers = []
        if self.initial_value != None:
            for i in range(len(self.units)):
                layers.append(Linear(units=self.units[i], 
                                     seed=self.seed, 
                                     initial_value=[self.initial_value[i*2],
                                                    self.initial_value[i*2+1]]))
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
        
        if self.embedding:
            s_x = self.state_emb(inputs['state'])
            a_x = self.action_emb(inputs['action'])
            t_x = self.step_emb(inputs['step'])
            
            x = tf.concat([s_x,a_x,t_x],axis=-1)
        else:
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