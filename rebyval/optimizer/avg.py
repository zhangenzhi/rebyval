import abc
import sys
import warnings

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

XLA_compile_ = True if sys.platform == 'win32' else None


class AverageOptimizerWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    def __init__(self, optimizer=None, name="AverageOptimizer", **kwargs):
        super(AverageOptimizerWrapper, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizer.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError('optimier is not type tf.keras.optimizers.Optimizer')

        self._optimizer = optimizer
        self._track_trackable(self._optimizer, "awg_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        return self._optimizer._prepare_local(var_device, var_dtype, apply_state)

    def apply_gradients(self, grad_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grad_and_vars, name, **kwargs)

    @abc.abstractmethod
    def average_op(self, var, average_var, local_apply_state):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var, apply_state):
        apply_state = apply_state or {}
        local_apply_state = apply_state.get((var.device, var.dtype.base_dtype))
        if local_apply_state is None:
            local_apply_state = self._fallback_apply_state(
                var.device, var.dtype.base_dtype
            )
        average_var = self.get_slot(var, "average")
        return self.average_op(var, average_var, local_apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad, var, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices
            )
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices
            )
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        assign_ops = []
        for var in var_list:
            try:
                assign_ops.append(
                    var.assign(
                        self.get_slot(var, "average"), use_locking=self._use_locking
                    )
                )
            except Exception as e:
                warnings.warn("Unable to assign average slot to {} : {}".format(var, e))
        return tf.group(assign_ops)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        return self._optimizer._set_hyper("learning_rate", lr)

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        return self._optimizer._set_hyper("learning_rate", learning_rate)


class SWA(AverageOptimizerWrapper):

    def __init__(
            self,
            optimizer,
            start_averaging=0,
            average_period=10,
            name="SWA",
            **kwargs
    ):
        super(SWA,self).__init__(optimizer, name, **kwargs)

        if average_period < 1:
            raise ValueError("average_period must >= 1")
        if start_averaging < 0:
            raise ValueError("start_averaging must >= 0")

        self._set_hyper("average_period", average_period)
        self._set_hyper("start_averaging", start_averaging)

    @tf.function(experimental_relax_shapes=True,
                 experimental_compile=XLA_compile_)
    def average_op(self, var, average_var, local_apply_state):
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)

        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period)
        )

        checkpoint = start_averaging + num_snapshots * average_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            average_value = (average_var * num_snapshots + var) / (num_snapshots + 1.0)
            return average_var.assign(average_value, use_locking=self._use_locking)
        return average_var

    def get_config(self):
        config = {
            "average_period": self._serialize_hyerparameter("average_period"),
            "start_averaging": self._serialize_hyerparameter("start_averaging")
        }
        base_config = super().get_config()

        return {**base_config, **config}
