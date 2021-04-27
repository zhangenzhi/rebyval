import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.base_trainer import BaseTrainer


class TargetTrainer(BaseTrainer):
    def __init__(self, trainer_args):
        super(TargetTrainer, self).__init__(trainer_args=trainer_args)

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'cifar10':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):
        try:
            x = self.train_iter.get_next()
            y = x.pop('label')
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
            _,self.valid_dataset,_ = self.reset_dataset()
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


class SurrogateTrainer(BaseTrainer):
    def __init__(self, trainer_args):
        super(SurrogateTrainer, self).__init__(trainer_args=trainer_args)

    @BaseTrainer.timer
    def during_train(self):
        pass

    @BaseTrainer.timer
    def during_valid(self):
        pass

    @BaseTrainer.timer
    def during_test(self):
        pass
