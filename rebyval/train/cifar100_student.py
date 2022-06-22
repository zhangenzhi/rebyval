from matplotlib.pyplot import sca
from sklearn import metrics
from tqdm import trange
import numpy as np
import tensorflow as tf

# others
import time
from rebyval.model.dnn import DNN
from rebyval.train.student import Student
from rebyval.train.utils import ForkedPdb
from rebyval.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar100Student(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar100Student, self).__init__(student_args, supervisor,id)

        self.action_space = np.random.uniform(low=1.0, high=1.0, size=100)
        self.index_min = 0
    
    def weightspace_loss(self, weights):
        # label
        flat_vars = []
        for var in weights:
            sum_reduce = tf.math.reduce_sum(var, axis= -1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(-1)))
        inputs = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        s_loss = self.supervisor(inputs)
        s_loss = tf.squeeze(s_loss)
        return s_loss
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels, train_step = 0, epoch=0, decay_factor=0.0001):
        
        step = train_step+epoch*self.dataloader.info['train_step']

        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        if train_step % 100 == 0:
            with tf.GradientTape() as tape_s:
                self.s_loss = self.weightspace_loss(self.model.trainable_variables)
            self.s_grad = tape_s.gradient(self.s_loss, self.model.trainable_variables)

        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        gradients = [(s/(1e-12 + tf.norm(s))*decay_factor + t/(1e-8 + tf.norm(t)))/2 for s,t in zip(self.s_grad,t_grad)]
        # gradients = [(s/(1e-12 + tf.norm(s)))*decay_factor+t for s,t in zip(self.s_grad,t_grad)]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        with self.logger.as_default():
            tf.summary.scalar("surrogate_loss", self.s_loss, step=step)
            # if step % self.dataloader.info['train_step'] == 0:
            #     tf.summary.histogram("t_gard_0", t_grad[0], step=step)
            #     tf.summary.histogram("s_gard_0", self.s_grad[0], step=step)
            
        self.mt_loss_fn.update_state(t_loss)
        
        return t_loss

                
                