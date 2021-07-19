import abc
import sys
import warnings

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

XLA_compile_ = True if sys.platform == 'win32' else None


class AverageOptimizerWrapper(tf.keras.optimizer.Optimizer, metaclass=abc.ABCMeta):
    def __init__(self, optimizer=None, name="AverageOptimizer", **kwargs):
        super(AverageOptimizerWrapper, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizer.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError('optimier is not type tf.keras.optimizers.Optimizer')
