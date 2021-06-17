import sys

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


XLA_compile_ = True if sys.platform == 'win32' else None

class LARS(tf.keras.optimizers.Optimizer):
    r'''
    LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS. Yang You, Igor Gitman, Boris Ginsburg.
    Update Rules:
        lr = lr_0 * (1-t/T)^2
        local_lr = (eta * ||w||) / (||g|| + beta * ||w||)
        v = momentum * v + lr * local_lr * (g + beta * w)
        w = w - v
    '''
    def __init__(self,
                 learning_rate=1.0,
                 momentum=0.9,
                 beta=0.0005,
                 eta=0.001,
                 name='LARS',
                 **kwargs):
        super(LARS, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('momentum', momentum)
        self._set_hyper('beta', beta)
        self._set_hyper('eta', eta)
        self.epsilon = 1e-8

    def _deduplicate_indexed_slices(self, values, indices):
        unique_indices, new_index_positions = array_ops.unique(indices)
        summed_values = math_ops.unsorted_segment_sum(
            values, new_index_positions,
            array_ops.shape(unique_indices)[0])
        return (summed_values, unique_indices)

    '''
    Prepare Tensor of assigned hyper-parameters.
    A build-in hyper-parameter tensor 'lr_t' will be deduced from learning rate schedules object.
    '''
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(LARS, self)._prepare_local(var_device, var_dtype, apply_state)

    # XXX Unclear how to use.
    # def set_weights(self, weight):
    #     pass

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'v')

    @tf.function(experimental_relax_shapes=True,
                 experimental_compile=XLA_compile_)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype

        learning_rate = apply_state[(var_device, var_dtype)]['lr_t']
        beta = self._get_hyper('beta')
        epsilon = self.epsilon
        eta = self._get_hyper('eta')
        momentum = self._get_hyper('momentum')

        var_norm = tf.norm(var, ord=2)
        grad_norm = tf.norm(var, ord=2)

        local_lr = var_norm / (grad_norm + beta * var_norm + epsilon)
        local_lr = local_lr * eta

        v = self.get_slot(var, 'v')
        v.assign(momentum * v + learning_rate * local_lr * (grad + beta * var))
        var.assign_sub(v)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                                 **kwargs):
        summed_grad, unique_indices = self._deduplicate_indexed_slices(
            values=grad, indices=indices)
        return self._resource_apply_sparse(summed_grad, handle, unique_indices,
                                           **kwargs)

    '''
    var:  embedding table: shape=[max_dim_size(e.g 2000), embedding_size(e.g 16)].
    grad: grad of used embedding table part: e.g shape=[1,16].
    indices: unique indices of embedding table slice: int vector tensor.
    # '''
    @tf.function(experimental_relax_shapes=True,
                 experimental_compile=XLA_compile_)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype

        learning_rate = apply_state[(var_device, var_dtype)]['lr_t']
        beta = self._get_hyper('beta')
        epsilon = self.epsilon
        eta = self._get_hyper('eta')
        momentum = self._get_hyper('momentum')

        grad_norm = tf.norm(grad, ord=2)

        # var_norm = tf.norm(var, ord=2)
        # local_lr = var_norm / (grad_norm + beta * var_norm + epsilon)
        # local_lr = local_lr * eta

        # TO-DO: layer-wise normalization instead of indices-wise normalization
        var_indiced = tf.gather(var, indices)
        var_indiced_norm = tf.norm(var_indiced, ord=2)
        local_lr_indiced = var_indiced_norm / (
            grad_norm + beta * var_indiced_norm + epsilon)
        local_lr_indiced = local_lr_indiced * eta

        local_lr = local_lr_indiced
        v = self.get_slot(var, 'v')
        v.assign(momentum * v + learning_rate * local_lr * beta * var)
        scaled_grad = learning_rate * local_lr * grad
        v.scatter_add(ops.IndexedSlices(scaled_grad, indices))
        var.assign_sub(v)

    def get_config(self):
        config = super(LARS, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'decay':
            self._serialize_hyperparameter('decay'),
            'momentum':
            self._serialize_hyperparameter('momentum'),
            'beta':
            self._serialize_hyperparameter('beta'),
            'eta':
            self._serialize_hyperparameter('eta'),
            'epsilon':
            self.epsilon,
        })
        return config
