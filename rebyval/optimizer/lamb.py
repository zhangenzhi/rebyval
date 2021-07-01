import sys

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

XLA_compile_ = True if sys.platform == 'win32' else None


class LAMB(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 weight_decay_rate=0.0,
                 exclude_from_weight_decay: list = None,
                 exclude_from_layer_adaptation: list = None,
                 name: str = 'LAMB',
                 **kwargs,
                 ):
        super().__init__(name, **kwargs)

        self._set_hyper("weight_decay_rate", weight_decay_rate)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)

        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slot(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

        for var in var_list:
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))

    def _resource_apply_dense(self):
        pass

    def _resource_apply_sparse(self):
        pass

    def get_config(self):
        pass

    def _do_use_weight_decay(self, param_name):
        pass

    def _do_layer_adaptation(self, param_name):
        pass

    def _get_variable_name(self, param_name):
        pass
