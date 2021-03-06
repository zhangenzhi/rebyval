import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.student import BaseTrainer


class TargetTrainer(BaseTrainer):
    def __init__(self, trainer_args, surrogate_model=None):
        super(TargetTrainer, self).__init__(trainer_args=trainer_args)
        self.surrogate_model = surrogate_model
        self.extra_metrics = {}
        if self.surrogate_model is not None:
            self.extra_metrics['v_loss'] = tf.Variable(0.0)
            self.extra_metrics['t_loss'] = tf.Variable(0.0)

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
                    .format(self.extra_metrics['v_loss'].numpy(), self.extra_metrics['t_loss'].numpy())
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
                v_loss = tf.reshape(v_loss, shape=())

                # verify v net
                loss = t_loss
                # loss = t_loss + v_loss * 0.001

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            self.metrics['train_loss'](loss)
            self.extra_metrics['t_loss'].assign(t_loss)
            self.extra_metrics['v_loss'].assign(v_loss)
        except:
            print_error("rebyval train step error")
            raise

    def run_with_weights_collect(self):
        if self.args['train_loop_control']['valid'].get('analyse'):
            self.run()
        else:
            print_error("analysis not open in Target Trainer")
            raise ("error")


class SurrogateTrainer(BaseTrainer):
    def __init__(self, trainer_args):
        super(SurrogateTrainer, self).__init__(trainer_args=trainer_args)

    def reset_dataset(self):
        dataset_args = self.args['dataloader']
        if self.args['dataloader']['name'] == 'dnn_weights':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset(format=dataset_args['format'])
            test_dataset = valid_dataset
            return train_dataset, valid_dataset, test_dataset

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _parse_tensor(self, x):
        parsed_tensors = {}
        for feat, tensor in x.items():
            batch_serilized_tensor = []
            for i in range(256):
                batch_serilized_tensor.append(tf.io.parse_tensor(tensor[i], tf.float32))
            parsed_tensors = tf.concat(batch_serilized_tensor, axis=0)
        return parsed_tensors

    @BaseTrainer.timer
    def during_train(self):

        try:
            x = self.train_iter.get_next()

        except:
            print_warning("during training dataset exception")
            try:
                self.epoch += 1
                self.train_dataset, _, _ = self.reset_dataset()
                self.train_iter = iter(self.train_dataset)
                x = self.train_iter.get_next()
            except:
                print_error("reset train iter failed")
                raise

        y = x.pop('valid_loss')
        if 'vars_length' in x.keys():
            x.pop('vars_length')
        x.pop('train_loss')

        try:
            if self.preprocess['name'] == 'sum_reduce':
                self._train_step_surrogate_sum_reduce(x, y)
            elif self.preprocess['name'] == 'l2_sum_reduce':
                self._train_step_surrogate_l2_sum_reduce(x, y)
            else:
                self._train_step_surrogate(x, y)
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
        if 'vars_length' in x_valid.keys():
            x_valid.pop('vars_length')

        if self.preprocess['name'] == 'sum_reduce':
            flat_vars = []
            for feat, tensor in x_valid.items():
                axis = tensor.shape.rank - 1
                compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs': flat_vars}
        elif self.preprocess['name'] == 'l2_sum_reduce':
            flat_vars = []
            for feat, tensor in x_valid.items():
                axis = tensor.shape.rank - 1
                if feat in ['vars_0', 'vars_1']:
                    axis = list(range(axis))
                    axis = [i + 1 for i in axis]
                    if len(axis) == 2:
                        compressed_tensor = tf.norm(tensor, axis=axis)
                    else:
                        compressed_tensor = tf.norm(tensor, axis=axis[0])
                else:
                    compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs': flat_vars}
        else:
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
            if 'vars_length' in x_test.keys():
                x_test.pop('vars_length')
            if self.preprocess['name'] == 'sum_reduce':
                flat_vars = []
                for feat, tensor in x_test.items():
                    axis = tensor.shape.rank - 1
                    compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                    flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
                flat_vars = tf.concat(flat_vars, axis=1)
                flat_input = {'inputs': flat_vars}
            elif self.preprocess['name'] == 'l2_sum_reduce':
                flat_vars = []
                for feat, tensor in x_test.items():
                    axis = tensor.shape.rank - 1
                    if feat in ['vars_0', 'vars_1']:
                        axis = list(range(axis))
                        axis = [i + 1 for i in axis]
                        if len(axis) == 2:
                            compressed_tensor = tf.norm(tensor, axis=axis)
                        else:
                            compressed_tensor = tf.norm(tensor, axis=axis[0])
                    else:
                        compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                    flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
                flat_vars = tf.concat(flat_vars, axis=1)
                flat_input = {'inputs': flat_vars}
            else:
                flat_vars = []
                for feat, tensor in x_test.items():
                    flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
                flat_vars = tf.concat(flat_vars, axis=1)
                flat_input = {'inputs': flat_vars}

            self._test_step(flat_input, y_test)
        except:
            self.test_flag = False

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step_surrogate(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}
        try:
            with tf.GradientTape() as tape:
                # import pdb
                # pdb.set_trace()
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
    def _train_step_surrogate_sum_reduce(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            axis = tensor.shape.rank - 1
            compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
            flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
        flat_vars = tf.concat(flat_vars, axis=1)
        flat_input = {'inputs': flat_vars}

        try:
            with tf.GradientTape() as tape:
                predictions = self.model(flat_input, training=True)
                loss = self.metrics['loss_fn'](labels, predictions)
                if self.args['model'].get('regularizer'):
                    re_loss = tf.constant(0.0)
                    for layer in self.model.dnn_layer:
                        re_loss += tf.math.reduce_sum(layer.losses)
                    loss += re_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            self.metrics['train_loss'](loss)
        except:
            print_error("train step error")
            raise

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step_surrogate_l2_sum_reduce(self, inputs, labels):
        flat_vars = []
        for feat, tensor in inputs.items():
            axis = tensor.shape.rank - 1
            if feat in ['vars_0', 'vars_1']:
                axis = list(range(axis))
                axis = [i+1 for i in axis]
                if len(axis) == 2:
                    compressed_tensor = tf.norm(tensor, axis=axis)
                else:
                    compressed_tensor = tf.norm(tensor,axis=axis[0])
            else:
                compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
            flat_vars.append(tf.reshape(compressed_tensor, shape=(tensor.shape[0], -1)))
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

    def run_with_refreshed_dataset(self):

        # re-prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
        self.dataloader = self._build_dataset()
        self.main_loop()
