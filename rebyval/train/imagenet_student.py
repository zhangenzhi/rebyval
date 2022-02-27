import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.student import BaseTrainer


class ImageNetTrainer(BaseTrainer):
    def __init__(self, trainer_args, surrogate_model=None):
        super(ImageNetTrainer, self).__init__(trainer_args=trainer_args)
        self.surrogate_model = surrogate_model
        self.extra_metrics = {}
        if self.surrogate_model is not None:
            self.extra_metrics['v_loss'] = tf.Variable(0.0)
            self.extra_metrics['t_loss'] = tf.Variable(0.0)

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def decode_image(self, image_raw, batch_size):
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
            if self.args['distribute']:
                train_dataset = self.mirrored_stragey.experimental_distribute_dataset(train_dataset)
                valid_dataset = self.mirrored_stragey.experimental_distribute_dataset(valid_dataset)
                test_dataset = self.mirrored_stragey.experimental_distribute_dataset(test_dataset)
            return train_dataset, valid_dataset, test_dataset

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

        y = x.pop('label')
        input = x['image_raw']

        try:
            if self.args['distribute']:
                self._distributed_train_step(input,y)
            else:
                self._train_step(input, y)
        except:
            print_error("during traning train_step exception")
            raise


    @BaseTrainer.timer
    def during_valid(self):
        try:
            x_valid = self.valid_iter.get_next()
            y_valid = x_valid.pop('label')
            input = x_valid['image_raw']
            try:
                if self.args['distribute']:
                    self._distributed_valid_step(input, y_valid)
                else:
                    self._valid_step(input, y_valid)
            except:
                print_error("during valid valid_step exception")
                raise

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
            input = x_test['image_raw']
            if self.args['distribute']:
                self._distributed_test_step(input, y_test)
            else:
                self._valid_step(input, y_test)
        except:
            self.test_flag = False


    def run_with_weights_collect(self):
        if self.args['train_loop_control']['valid'].get('analyse'):
            self.run()
        else:
            print_error("analysis not open in Target Trainer")
            raise ("error")
