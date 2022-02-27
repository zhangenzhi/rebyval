import tensorflow as tf
from rebyval.train.supervisor import Supervisor

from rebyval.train.utils import get_scheduler, prepare_dirs
from rebyval.tools.utils import calculate_auc, write_log, print_green, print_error, print_normal

class Cifar10Supervisor(Supervisor):
    def __init__(self, supervisor_args):
        super().__init__(supervisor_args)

    def __call__(self, inputs):
        flat_vars = []
        for feat, tensor in inputs.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}
        return self.model(flat_input)

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(flat_input, training=True)
                loss = self.metrics['loss_fn'](labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            self.metrics['train_loss'](loss)
        except:
            print_error("train step error")
            raise 

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(flat_input, training=True)
                loss = self.metrics['loss_fn'](labels, predictions)
        except:
            print_error("valid step error.")
            raise 

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(flat_input, training=True)
                loss = self.metrics['loss_fn'](labels, predictions)
        except:
            print_error("test step error.")
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