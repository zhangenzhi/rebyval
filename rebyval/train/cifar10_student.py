
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

                
                
        