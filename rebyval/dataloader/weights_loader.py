import os
import random
import tensorflow as tf

from rebyval.tools.utils import get_yml_content
from rebyval.dataloader.utils import glob_tfrecords
from rebyval.dataloader.base_dataloader import BaseDataLoader


class DNNWeightsLoader(BaseDataLoader):

    def __init__(self, dataloader_args):
        super(DNNWeightsLoader, self).__init__(dataloader_args=dataloader_args)
        
    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path,name)
        yaml_feature_config = get_yml_content(yaml_path)
        
        self.info = {'epochs': self.dataloader_args['epochs'],
                     'total_samples': yaml_feature_config['total_samples'],
                     'train_samples': int(yaml_feature_config['total_samples']*0.6),
                     'test_samples': int(yaml_feature_config['total_samples']*0.2), 
                     'valid_samples': int(yaml_feature_config['total_samples']*0.2),
                     'train_step': int(yaml_feature_config['total_samples']*0.6 / self.dataloader_args['batch_size']),
                     'valid_step': int(yaml_feature_config['total_samples']*0.2 / self.dataloader_args['batch_size']),
                     'test_step': int(yaml_feature_config['total_samples']*0.2 / self.dataloader_args['batch_size'])
            }
        
        feature_config = {
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'list', "length": yaml_feature_config['vars_length']['value'], "dtype": tf.string},
            }

        return feature_config

    def _make_analyse_tensor_describs(self, feature_config=None):

        feature_describs = {}
        for feature, info in feature_config.items():
            if info['type'] == 'list':
                for i in range(info["length"]):
                    feature_type = tf.io.FixedLenFeature([], info["dtype"])
                    feature_describs[feature +
                                             "_{}".format(i)] = feature_type
                info_type = tf.io.FixedLenFeature([], tf.int64)
                feature_describs[feature + "_length"] = info_type
            elif info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        return feature_describs

    def _load_analyse_tensor_from_tfrecord(self, filelist, feature_config):

        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=16),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=16,
            deterministic=False)

        # raw_analyse_dataset = raw_analyse_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)

        # feature_config = self._feature_config_parse(self.dataloader_args[''])
        analyse_feature_describ = self._make_analyse_tensor_describs(
            feature_config = feature_config)

        def _parse_weights_function(example_proto):
            example = tf.io.parse_example(
                example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    # parsed_single_example = []
                    # for i in range(self.dataloader_args['batch_size']):
                    #     parsed_single_example.append(tf.io.parse_tensor(example[feat][i], out_type=tf.float32))
                    # parsed_example[feat] = parsed_single_example
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_weights_function,
                                                         num_parallel_calls=16, deterministic=True)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset
    
    def load_dataset(self, format=None):

        filelist = glob_tfrecords(
            self.dataloader_args['path'], glob_pattern='*.tfrecords')
        feature_config = self._feature_config_parse(self.dataloader_args['path'], 
                                                    name='feature_configs.yaml')
        
        full_dataset = self._load_analyse_tensor_from_tfrecord(filelist=filelist,
                                                               feature_config=feature_config)
        
        train_dataset = full_dataset.take(self.info['train_samples'])
        
        valid_dataset = full_dataset.skip(self.info['train_samples'])
        valid_dataset = valid_dataset.take(self.info['valid_samples'])
        
        test_dataset = valid_dataset.skip(self.info['valid_samples'])
        test_dataset = valid_dataset.take(self.info['test_samples'])
        
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        train_dataset = train_dataset.repeat(self.info["epochs"])
        
        valid_dataset = valid_dataset.take(self.info['valid_samples'])
        valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])
        valid_dataset = valid_dataset.repeat(self.info["epochs"])
        
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'])

        return train_dataset, valid_dataset, test_dataset
    
# ds = iter(full_dataset)
# for i in range(1000):
#     ds.get_next()
#     print(i)