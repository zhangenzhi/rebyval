import tensorflow as tf
from rebyval.dataloader.utils import glob_tfrecords
from rebyval.dataloader.base_dataloader import BaseDataLoader


class DnnWeightsLoader(BaseDataLoader):

    def __init__(self, dataloader_args):
        super(DnnWeightsLoader, self).__init__(dataloader_args=dataloader_args)

    def _make_analyse_describs(self, num_trainable_variables, analyse_feature=None):
        if analyse_feature == None:
            analyse_feature = {
                'train_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'list', "length": num_trainable_variables, "dtype": tf.string},
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
            else:
                raise ("no such type to describe")
        return analyse_feature_describs

    def _load_analyse_from_tfrecord(self, filelist, num_trainable_variables):

        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=16),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=16,
            deterministic=False)

        raw_analyse_dataset = raw_analyse_dataset.batch(self.dataloader_args['batch_size'],drop_remainder=True)

        analyse_feature_describ = self._make_analyse_describs(
            num_trainable_variables)

        def _parse_analyse_function(example_proto):
            example = tf.io.parse_example(
                example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_single_example = []
                    for i in range(self.dataloader_args['batch_size']):
                        parsed_single_example.append(tf.io.parse_tensor(example[feat][i], out_type=tf.float32))
                    parsed_example[feat] = parsed_single_example
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_analyse_function,
                                                         num_parallel_calls=64, deterministic=False)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset

    def load_dataset(self):

        filelist = glob_tfrecords(
            self.dataloader_args['datapath'], glob_pattern='*.tfrecords')
        print(filelist)
        train_dataset_size = int(len(filelist)*1000*0.75/self.dataloader_args['batch_size'])


        fulldataset = self._load_analyse_from_tfrecord(filelist=filelist,
                                                   num_trainable_variables=self.dataloader_args[
                                                       'num_trainable_variables'])
        fulldataset = fulldataset.shuffle(len(filelist)*10)

        train_dataset = fulldataset.take(train_dataset_size).cache()
        valid_dataset = fulldataset.skip(train_dataset_size).cache()


        return train_dataset.repeat(-1), valid_dataset.repeat(-1), valid_dataset
