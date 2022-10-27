from functools import reduce
import os
from tqdm import trange
import numpy as np
import tensorflow as tf

# others
import time
from rebyval.train.student import Student
from rebyval.train.utils import ForkedPdb
from rebyval.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar10Student(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar10Student, self).__init__(student_args, supervisor,id)

        self.action_space = np.random.uniform(low=1.0, high=1.0, size=100)
        self.index_min = 0
    
    def weight_reduce(self, weights):
        flat_vars = []
        for var in weights:
            sum_reduce = tf.math.reduce_sum(var, axis= -1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(-1)))
        reduced_w = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        return reduced_w
    
    def weight_flatten(self, weights):
        flat_vars = []
        for var in weights:
            flat_vars.append(tf.reshape(var, shape=(-1)))
        flatten_w = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        return flatten_w
    
    def weightspace_loss(self, weights, format="flatten"):
        # label
        if format == "sum_reduce":
            inputs = self.weight_reduce(weights)
        else:
            inputs = self.weight_flatten(weights)
        s_loss = self.supervisor(inputs)
        s_loss = tf.squeeze(s_loss)
        return s_loss

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _reval_train_step(self, inputs, labels, decay_factor=0.1):
        
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
            self.train_metrics.update_state(labels, predictions)
        
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
    
        with tf.GradientTape() as tape_s:
            s_loss = self.weightspace_loss(self.model.trainable_variables)
        s_grad = tape_s.gradient(s_loss, self.model.trainable_variables)
        
        gradients = [(s/(1e-12 + tf.norm(s)))*decay_factor + t for s,t in zip(s_grad,t_grad)]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        self.mt_loss_fn.update_state(t_loss)
        
        return t_loss, s_loss
    
    def _ireval_train_step(self, inputs, labels, decay_factor=0.1, format="flatten"):
        
        # train loss
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
            self.train_metrics.update_state(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)

        # exp vloss
        # with tf.GradientTape() as tape_s:
        s_loss = self.weightspace_loss(self.model.trainable_variables, format = format)
        # s_grad = tape_s.gradient(s_loss, self.model.trainable_variables)
        s_grad = self.exp_grad
        
        # true vloss
        # v_preds = self.model(v_inputs, training=False)
        # v_loss = self.loss_fn(v_labels, v_preds)
        
        gradients = [s*decay_factor + t for s,t in zip(s_grad, t_grad)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # online update
        # self.update_supervisor(self.model.trainable_variables, v_loss)
        
        self.mt_loss_fn.update_state(t_loss)
        
        return t_loss, s_loss
                
                
        