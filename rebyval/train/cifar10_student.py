import tensorflow as tf

# others
from rebyval.train.student import Student
from rebyval.train.utils import get_scheduler, prepare_dirs
from rebyval.tools.utils import calculate_auc, write_log, print_green, print_error, print_normal

class Cifar10Student(Student):
    
    def __init__(self, student_args, supervisor = None):
        super(Cifar10Student, self).__init__(student_args, supervisor)


    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        except:
            print_error("train step error")
            raise

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                s_loss = self.supervisor(self.model.trainable_variables)
                t_loss = self.loss_fn(labels, predictions)
                loss = t_loss + s_loss
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        except:
            print_error("rebyval train step error")
            raise

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        except:
            print_error("valid step error")
            raise

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        except:
            print_error("valid step error")
            raise

    def train(self):
        # parse train loop control args
        train_loop_control_args = self.args['train_loop_control']
        train_args = train_loop_control_args['train']

        # dataset train, valid, test
        epoch = 0
        step = 0
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        # metrics reset
        metric_name = self.args['metrics']['name']
        self.metrics[metric_name].reset_states()

        # train, valid, test, write to tfrecords