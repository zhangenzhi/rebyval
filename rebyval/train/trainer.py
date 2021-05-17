import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.base_trainer import BaseTrainer


class TargetTrainer(BaseTrainer):
    def __init__(self, trainer_args, surrogate_model=None):
        super(TargetTrainer, self).__init__(trainer_args=trainer_args)
        self.surrogate_model = surrogate_model
        self.extra_metrics = {}
        if self.surrogate_model is not None:
            self.extra_metrics['v_loss'] = tf.keras.metrics.Mean(name='surrogate_loss')
            self.extra_metrics['t_loss'] = tf.keras.metrics.Mean(name='target_loss')

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'cifar10':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):
        try:
            x = self.train_iter.get_next()
            y = x.pop('label')
            if self.surrogate_model is not None:
                self._train_step_rebyval(x, y)
                extra_train_msg = '[Extra Status]: surrogate loss={:04f}, target loss={:.4f}' \
                       .format(self.extra_metrics['v_loss'].result, self.extra_metrics['t_loss'].result()) 
                print_green(extra_train_msg)
            else:
                self._train_step(x, y)
        except:
            print_warning("during traning exception")
            self.epoch += 1
            self.train_dataset, _, _ = self.reset_dataset()
            self.train_iter = iter(self.train_dataset)

    @BaseTrainer.timer
    def during_valid(self):
        try:
            x_valid = self.valid_iter.get_next()
            y_valid = x_valid.pop('label')
            self._valid_step(x_valid, y_valid)

        except:
            print_warning("during validation exception")
            _, self.valid_dataset, _ = self.reset_dataset()
            self.valid_iter = iter(self.valid_dataset)

    @BaseTrainer.timer
    def during_test(self):
        print('[Testing Status]: Test Step: {:04d}'.format(self.test_step))

        try:
            x_test = self.test_iter.get_next()
            y_test = x_test.pop('label')
            self._test_step(x_test, y_test)
        except:
            self.test_flag = False

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step_rebyval(self, inputs, labels):
        try:
            v_inputs = {'inputs': None}
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                t_loss = self.metrics['loss_fn'](labels, predictions)

                weights_flat = [tf.reshape(w, (1, -1)) for w in self.model.trainable_variables]
                v_inputs['inputs'] = tf.concat(weights_flat, axis=1)
                v_loss = self.surrogate_model(v_inputs)

                loss = t_loss + v_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            self.metrics['train_loss'](loss)
            self.extra_metrics['t_loss'](t_loss)
            self.extra_metrics['v_loss'](v_loss)
        except:
            print_error("rebyval train step error")
            raise

    def run_with_weights_collect(self):
        if self.args['train_loop_control']['valid']['analyse']:
            self.run()
        else:
            print_error("analysis not open in Target Trainer")
            raise ("error")


class SurrogateTrainer(BaseTrainer):
    def __init__(self, trainer_args):
        super(SurrogateTrainer, self).__init__(trainer_args=trainer_args)

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'dnn_weights':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):

        try:
            x = self.train_iter.get_next()
        except:
            print_warning("during traning dataset exception")
            try:
                self.epoch += 1
                self.train_dataset, _, _ = self.reset_dataset()
                self.train_iter = iter(self.train_dataset)
                x = self.train_iter.get_next()
            except:
                print_error("reset train iter failed")
                raise

        # y = x.pop('valid_loss')
        # x.pop('train_loss')
        # x.pop('vars_length')
        # flat_vars = []
        # for feat, tensor in x.items():
        #     flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        # # import pdb
        # # pdb.set_trace()
        # flat_vars = tf.concat(flat_vars, axis=1)
        # flat_input = {'inputs': flat_vars}

        try:
            # self._train_step(flat_input, y)
            print("good luck")
        except:
            print_error("during traning train_step exception")
            raise

    @BaseTrainer.timer
    def during_valid(self):

        try:
            x_valid = self.valid_iter.get_next()
        except:
            print_warning("during validatioin dataset exception")
            try:
                _, self.valid_dataset, _ = self.reset_dataset()
                self.valid_iter = iter(self.valid_dataset)
                x_valid = self.valid_iter.get_next()
            except:
                print_error("reset train iter failed")
                raise

        y_valid = x_valid.pop('valid_loss')
        x_valid.pop('train_loss')
        x_valid.pop('vars_length')
        flat_vars = []
        for feat, tensor in x_valid.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}

        try:
            self._valid_step(flat_input, y_valid)
        except:
            print_error("during validation valid_step exception")
            raise


    @BaseTrainer.timer
    def during_test(self):

        print('[Testing Status]: Test Step: {:04d}'.format(self.test_step))

        try:
            x_test = self.test_iter.get_next()
            y_test = x_test.pop('valid_loss')
            x_test.pop('train_loss')
            x_test.pop('vars_length')
            flat_vars = []
            for feat, tensor in x_test.items():
                flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs': flat_vars}

            self._test_step(flat_input, y_test)
        except:
            self.test_flag = False

    def run_main_loop(self):

        # re-prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
        self.dataloader = self._build_dataset()
        self.main_loop()