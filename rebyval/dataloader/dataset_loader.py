from json.tool import main
import os
import random
from unicodedata import name
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from rebyval.dataloader.utils import glob_tfrecords, normalization
from rebyval.dataloader.base_dataloader import BaseDataLoader

class MinistDataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super().__init__(dataloader_args)
        self.info = {'train_size':50000,'test_size':10000,'image_size':[28, 28, 1],
                'train_step': int(50000/dataloader_args['batch_size']),
                'valid_step': int(10000/dataloader_args['batch_size']),
                'test_step': int(10000/dataloader_args['batch_size']),
                'epochs': dataloader_args['epochs']}
    
    def load_dataset(self, epochs=-1, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        x_train = np.reshape(x_train, [60000,28,28,1])
        x_test = np.reshape(x_test, [10000,28,28,1])
        if self.dataloader_args['da']:
            x_train,x_test = normalization(x_train, x_test)
        
        #on-hot
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        data_augmentation = tf.keras.Sequential([
                    preprocessing.RandomContrast(0.1),
                    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                    preprocessing.RandomCrop(28, 28),
                    preprocessing.RandomZoom(0.1)
                    ])

        full_size = len(x_train)
        test_size = len(x_test)
        
        train_size = int(1.0 * full_size)
        valid_size = int(0.5 * test_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        full_dataset = full_dataset.shuffle(full_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        # data augmentation
        if self.dataloader_args['da']:
            train_dataset = train_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs']),'labels': x['labels']}, num_parallel_calls=16)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(epochs)


        # valid_dataset = full_dataset.skip(train_size)
        # valid_dataset = valid_dataset.take(valid_size).repeat(epochs)
        # valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        test_dataset = test_dataset.shuffle(test_size)
        # valid_dataset = test_dataset.take(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        # test_dataset = test_dataset.skip(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size']).repeat(epochs)
        valid_dataset = test_dataset

        return train_dataset, valid_dataset, test_dataset

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(Cifar10DataLoader, self).__init__(dataloader_args=dataloader_args)
        self.info = {'train_size':50000,'test_size':10000,'image_size':[32,32,3],
                     'train_step': int(50000/dataloader_args['batch_size']),
                     'valid_step': int(10000/dataloader_args['batch_size']),
                     'test_step': int(10000/dataloader_args['batch_size']),
                     'epochs': dataloader_args['epochs']}

    def load_dataset(self, epochs=-1, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        if self.dataloader_args['da']:
            x_train,x_test = normalization(x_train, x_test)
        
        # one-hot
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        data_augmentation = tf.keras.Sequential([
                            preprocessing.RandomFlip(mode="horizontal"),
                            preprocessing.RandomContrast(0.1),
                            preprocessing.RandomTranslation(height_factor=0.3, width_factor=0.3),
                            preprocessing.RandomCrop(32, 32),
                            preprocessing.RandomRotation(factor=(-0.2, 0.2)),
                            preprocessing.RandomZoom(0.2)
                            ])

        full_size = len(x_train)
        test_size = len(x_test)
        
        train_size = int(1.0 * full_size)
        valid_size = int(0.5 * test_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        full_dataset = full_dataset.shuffle(train_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        # data augmentation
        if self.dataloader_args['da']:
            train_dataset = train_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs'], training=True),'labels': x['labels']}, num_parallel_calls=16)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(epochs)

        # valid_dataset = full_dataset.skip(train_size)
        # valid_dataset = valid_dataset.take(valid_size).repeat(epochs)
        # valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        # all 1w test
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size']).repeat(-1)
        valid_dataset = test_dataset
        
        # test_dataset = test_dataset.shuffle(test_size)
        # valid_dataset = test_dataset.take(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        # test_dataset = test_dataset.skip(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)

        return train_dataset, valid_dataset, test_dataset

class Cifar100DataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(Cifar100DataLoader, self).__init__(dataloader_args=dataloader_args)
        self.info = {'train_size':50000,'test_size':10000,'image_size':[32,32,3],
                     'train_step': int(50000/dataloader_args['batch_size']),
                     'valid_step': int(10000/dataloader_args['batch_size']),
                     'test_step': int(10000/dataloader_args['batch_size']),
                     'epochs': dataloader_args['epochs']}

    def load_dataset(self, epochs=-1, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        if self.dataloader_args['da']:
            x_train,x_test = normalization(x_train, x_test)
        
        #one-hot
        y_train = tf.keras.utils.to_categorical(y_train, 100)
        y_test = tf.keras.utils.to_categorical(y_test, 100)
        
        data_augmentation = tf.keras.Sequential([
                    preprocessing.RandomFlip(mode="horizontal"),
                    preprocessing.RandomContrast(0.1),
                    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                    preprocessing.RandomCrop(32, 32),
                    preprocessing.RandomRotation(factor=(-0.1, 0.1)),
                    preprocessing.RandomZoom(0.1)
                    ])

        full_size = len(x_train)
        test_size = len(x_test)
        
        train_size = int(1.0 * full_size)
        valid_size = int(0.5 * test_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        full_dataset = full_dataset.shuffle(full_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        # data augmentation
        if self.dataloader_args['da']:
            train_dataset = train_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs']),'labels': x['labels']}, num_parallel_calls=16)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(epochs)

        # valid_dataset = full_dataset.skip(train_size)
        # valid_dataset = valid_dataset.take(valid_size).repeat(epochs)
        # valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        # all 1w test
        test_dataset = test_dataset.shuffle(test_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        valid_dataset = test_dataset
        
        # test_dataset = test_dataset.shuffle(test_size)
        # valid_dataset = test_dataset.take(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        # test_dataset = test_dataset.skip(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)

        return train_dataset, valid_dataset, test_dataset


class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, dataloader_args):
        super(ImageNetDataLoader, self).__init__(dataloader_args=dataloader_args)

    def _make_imagenet_describs(self, analyse_feature=None):
        if analyse_feature == None:
            analyse_feature = {
                'height': {"type": 'value', "length": 1, "dtype": tf.int64},
                'width': {"type": 'value', "length": 1, "dtype": tf.int64},
                'depth': {"type": 'value', "length": 1, "dtype": tf.int64},
                'label': {"type": 'value', "length": 1, "dtype": tf.int64},
                'image_raw': {"type": 'var_value', "length": 1, "dtype": tf.string},
            }

        analyse_feature_describs = {}
        for feature, info in analyse_feature.items():
            if info['type'] == 'list':
                for i in range(info["length"]):
                    feature_type = tf.io.FixedLenFeature([], info["dtype"])
                    analyse_feature_describs[feature +
                                             "_{}".format(i)] = feature_type
                info_type = tf.io.FixedLenFeature([], tf.int64)
                analyse_feature_describs[feature + "_length"] = info_type
            elif info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                analyse_feature_describs[feature] = feature_type
            elif info['type'] == 'var_value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                analyse_feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        return analyse_feature_describs

    def _load_train_imagenet_from_tfrecord(self, filelist):
        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=28),
            block_length=56,
            cycle_length=28,
            num_parallel_calls=28,
            deterministic=False)

        raw_analyse_dataset = raw_analyse_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)
        analyse_feature_describ = self._make_imagenet_describs()

        def _parse_analyse_function(example_proto):
            example = tf.io.parse_example(example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_single_example = []
                    for i in range(self.dataloader_args['batch_size']):
                        parsed_analyse_image = tf.io.decode_jpeg(example[feat][i], channels=3)
                        resized_image = tf.image.resize(parsed_analyse_image, [224, 224])
                        resized_image = tf.image.random_crop(resized_image, [224, 224, 3])
                        resized_image = tf.image.random_brightness(resized_image, 0.1)
                        resized_image = tf.image.random_flip_left_right(resized_image)
                        resized_image = tf.cast(resized_image, tf.float32)
                        resized_image = (resized_image / 127.5) - 1.0
                        resized_image = tf.expand_dims(resized_image, axis=0)
                        parsed_single_example.append(resized_image)
                    parsed_single_example = tf.concat(parsed_single_example, axis=0)
                    parsed_example[feat] = parsed_single_example
                else:
                    parsed_example[feat] = example[feat]
            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_analyse_function,
                                                         num_parallel_calls=28, deterministic=False)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset

    def _load_test_imagenet_from_tfrecord(self, filelist):
        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=28),
            block_length=56,
            cycle_length=28,
            num_parallel_calls=28,
            deterministic=False)

        raw_analyse_dataset = raw_analyse_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)
        analyse_feature_describ = self._make_imagenet_describs()

        def _parse_analyse_function(example_proto):
            example = tf.io.parse_example(example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_single_example = []
                    for i in range(self.dataloader_args['batch_size']):
                        parsed_analyse_image = tf.io.decode_jpeg(example[feat][i], channels=3)
                        resized_image = tf.image.resize(parsed_analyse_image, [224, 224])
                        resized_image = tf.cast(resized_image, tf.float32)
                        resized_image = (resized_image / 127.5) - 1.0
                        resized_image = tf.expand_dims(resized_image, axis=0)
                        parsed_single_example.append(resized_image)
                    parsed_single_example = tf.concat(parsed_single_example, axis=0)
                    parsed_example[feat] = parsed_single_example
                else:
                    parsed_example[feat] = example[feat]
            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_analyse_function,
                                                         num_parallel_calls=28, deterministic=False)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset

    def load_dataset(self, format=None):

        train_dataset_path = os.path.join(self.dataloader_args['datapath'], 'train_shuffled')
        valid_dataset_path = os.path.join(self.dataloader_args['datapath'], 'valid_shuffled')

        train_filelist = glob_tfrecords(train_dataset_path, glob_pattern='*.tfrecords')
        test_filelist = valid_filelist = glob_tfrecords(valid_dataset_path, glob_pattern='*.tfrecords')
        if self.dataloader_args.get('sample_of_curves'):
            train_filelist = train_filelist[(len(train_filelist) - self.dataloader_args['sample_of_curves']):]
            if train_filelist == []:
                raise ('no files included.')

        print(len(train_filelist), train_filelist)
        print(len(valid_filelist), valid_filelist)
        print(len(test_filelist), test_filelist)

        train_dataset = self._load_train_imagenet_from_tfrecord(filelist=train_filelist)

        valid_dataset = self._load_test_imagenet_from_tfrecord(filelist=valid_filelist)

        test_dataset = self._load_test_imagenet_from_tfrecord(filelist=test_filelist)

        train_dataset = train_dataset.repeat(-1)
        valid_dataset = valid_dataset.repeat(-1)

        return train_dataset, valid_dataset, test_dataset
