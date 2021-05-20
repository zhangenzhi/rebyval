import tensorflow as tf
from rebyval.dataloader.base_dataloader import BaseDataLoader


class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(Cifar10DataLoader, self).__init__(dataloader_args=dataloader_args)

    def load_dataset(self, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        full_train_size = len(x_train)
        valid_size = int(0.2 * full_train_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'label': y_train})

        train_dataset = full_dataset.batch(self.dataloader_args['batch_size'])
        train_dataset = train_dataset.shuffle(self.dataloader_args['batch_size'])
        train_dataset = train_dataset.repeat(-1)

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'label': y_test})
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'])
        valid_dataset = test_dataset.repeat(-1)

        return train_dataset, valid_dataset, test_dataset


class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(ImageNetDataLoader, self).__init__(dataloader_args=dataloader_args)

    def load_dataset(self, format=None):
        pass
