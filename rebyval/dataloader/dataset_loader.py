import random
import tensorflow as tf
from rebyval.dataloader.utils import glob_tfrecords
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

    def _load_imagenet_from_tfrecord(self, filelist):
        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=16),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=16,
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
                        resized_image = tf.image.resize(parsed_analyse_image, [256, 256])
                        resized_image = tf.expand_dims(resized_image, axis=0)
                        parsed_single_example.append(resized_image)
                    parsed_single_example = tf.concat(parsed_single_example, axis=0)
                    parsed_example[feat] = parsed_single_example
                else:
                    parsed_example[feat] = example[feat]
            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_analyse_function,
                                                         num_parallel_calls=64, deterministic=False)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset

    def load_dataset(self, format=None):

        filelist = glob_tfrecords(
            self.dataloader_args['datapath'], glob_pattern='*.tfrecords')

        train_filelist = valid_filelist = test_filelist = []
        if self.dataloader_args.get('sample_of_curves'):

            train_filelist = filelist[(len(filelist) - self.dataloader_args['sample_of_curves']):]
            test_filelist = [f for f in filelist if f not in train_filelist]
            valid_filelist = random.sample(test_filelist, 5)
            test_filelist = valid_filelist

            if train_filelist == []:
                raise ('no files included.')

        print(len(train_filelist), train_filelist)
        print(len(valid_filelist), valid_filelist)
        print(len(test_filelist), test_filelist)

        train_dataset = self._load_imagenet_from_tfrecord(filelist=train_filelist)

        valid_dataset = self._load_imagenet_from_tfrecord(filelist=valid_filelist)

        test_dataset = self._load_imagenet_from_tfrecord(filelist=test_filelist)

        train_dataset = train_dataset.shuffle(len(train_filelist) * 100).cache().repeat(-1)
        valid_dataset = valid_dataset.cache().repeat(-1)

        return train_dataset, valid_dataset, test_dataset
