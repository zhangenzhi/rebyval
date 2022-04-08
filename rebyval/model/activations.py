
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation

def neu(x, w):
    pass

keras.utils.get_custom_objects().update({"neu":Activation(neu)})

if __name__ == '__main__':
    from .dnn import Linear
    x = tf.initializers.random_uniform()(shape=(32,10))
    layer = Linear(units=10)
    
    x = layer(x)
    act = keras.activations.get("neu")