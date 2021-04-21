import os
import time
import tensorflow as tf

from functools import wraps

class Trainer:
    def __init__(self, args):
        self.args = self._prepare_logdir(args)

        # create timer collection
        member_func = [
            func for func in dir(self)
            if callable(getattr(self, func)) and not func.startswith("__")
        ]
        expanded_func = ["cumulate_" + func for func in member_func]
        self.timer_dict = dict(zip(expanded_func, [0] * len(expanded_func)))

    @classmethod
    def timer(cls, func):
        @wraps(func)
        def wrapper(self, *arg, **kwds):
            start_time = time.time()
            res = func(self, *arg, **kwds)
            end_time = time.time()
            cost_time = end_time - start_time
            self.timer_dict[func.__name__] = cost_time
            self.timer_dict["cumulate_" + func.__name__] += cost_time
            return res

        return wrapper

    def _prepare_logdir(self, args):
        args.log_file = os.path.join(args.log_dir, args.log_file)
        args.model_dir = os.path.join(args.log_dir, 'models')
        args.tensorboard_dir = os.path.join(args.log_dir, 'tensorboard')

        auto_makedirs(args.model_dir)
        auto_makedirs(args.tensorboard_dir)

        return args

    def _build_enviroment(self):
        pass

    def _build_dataset(self):
        pass

    def _build_model(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_loss(self, loss=None):
        pass

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.metrics['loss_fn'](labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)


        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.metrics['train_loss'](loss)
        self.metrics['train_accuracy'](labels, tf.sigmoid(predictions))

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _infer_step(self, inputs, labels, name=None):

        predictions = self.model(inputs, training=False)
        v_loss = self.metrics['loss_fn'](labels, predictions)

        if name != None:
            self.metrics['{}_loss'.format(name)](v_loss)
            self.metrics['{}_accuracy'.format(name)](labels, predictions)

        return predictions

    def run(self):
        # set enviroment
        self._build_enviroment()

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
        self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.metrics = self._build_losses()

        # train
        train_msg = self.train()
