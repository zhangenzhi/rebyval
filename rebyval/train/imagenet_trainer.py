import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.base_trainer import BaseTrainer



class ImageNetTargetTrainer(BaseTrainer):
    def __init__(self, trainer_args, surrogate_model=None):
        super(ImageNetTargetTrainer, self).__init__(trainer_args=trainer_args)
        self.surrogate_model = surrogate_model
        self.extra_metrics = {}
        if self.surrogate_model is not None:
            self.extra_metrics['v_loss'] = tf.Variable(0.0)
            self.extra_metrics['t_loss'] = tf.Variable(0.0)

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def decode_image(image_raw, batch_size):
        decoded_image_batch = []
        for i in range(batch_size):
            decoded_image = tf.io.decode_jpeg(image_raw[i], channels=3)
            resized_image = tf.image.resize(decoded_image, [256, 256])
            resized_image = tf.expand_dims(resized_image, axis=0)
            decoded_image_batch.append(resized_image)
        decoded_image = tf.concat(decoded_image_batch, axis=0)
        return decoded_image

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'imagenet':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):
        try:
            x = self.train_iter.get_next()

            import pdb
            pdb.set_trace()

            y = x.pop('label')
            x = self.decode_image(x['image_raw'])
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
            x_valid = self.decode_image(x_valid['image_raw'])
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
            x_test = self.decode_image(x_test['image_raw'])
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
