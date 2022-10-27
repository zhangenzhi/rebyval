import os
import random
import tensorflow as tf

from rebyval.tools.utils import get_yml_content, print_green
from rebyval.dataloader.utils import glob_tfrecords
from rebyval.dataloader.base_dataloader import BaseDataLoader
from rebyval.dataloader.weights_loader import WeightsLoader

class DNNWeightsLoader(WeightsLoader):

    def __init__(self, dataloader_args):
        super(DNNWeightsLoader, self).__init__(dataloader_args=dataloader_args)
    
    def buffer_policy(self, filelist, new_students=[]):
        # random exp
        if self.dataloader_args['exp'] == "random":
            if len(filelist) > self.dataloader_args['replay_window']:
                random.shuffle(filelist)
                filelist = list(set(filelist) - set(new_students))
                past = [filelist.pop() for _ in range(self.dataloader_args['replay_window'] - len(new_students))]
                filelist = new_students + past
            print("filelist length: {}".format(len(filelist)))
        elif self.dataloader_args['exp'] == "decay":
            #decay exp
            if len(self.replay_buffer) > self.dataloader_args['replay_window']:
                for i in range(len(new_students)):
                    self.replay_buffer.pop(0)
                self.replay_buffer += new_students
            else:
                self.replay_buffer += new_students
            filelist = self.replay_buffer
        return filelist
        
    def load_dataset(self, new_students=[]):
        # load samples to mem
        
        print_green("weight_space_path:{}".format(self.dataloader_args['path']))
        filelist = glob_tfrecords(self.dataloader_args['path'], glob_pattern='*.tfrecords')
        filelist = self.buffer_policy(filelist, new_students=new_students)
        print_green("students:{}".format(filelist))
        self.info = self.get_info_inference(num_of_students=len(filelist),sample_per_student=self.info['sample_per_student'])
        full_dataset = self._load_tensor_from_tfrecord(filelist=filelist, feature_config=self.feature_config)
        full_dataset = full_dataset.shuffle(self.info['total_samples'])
        
        iter_ds = iter(full_dataset)
        
        train_samples = [iter_ds.get_next() for _ in range(self.info['train_samples']) ]
        train_x = [train_samples[i]['vars'] for i in range(len(train_samples))]
        train_y = [train_samples[i]['valid_loss'] for i in range(len(train_samples))]
        train_dataset = tf.data.Dataset.from_tensor_slices({'vars':train_x,'valid_loss':train_y})
        train_dataset = train_dataset.repeat(self.info["epochs"])
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        
        
        test_samples = [iter_ds.get_next() for _ in range(self.info['test_samples']) ]
        test_x = [test_samples[i]['vars'] for i in range(len(test_samples))]
        test_y = [test_samples[i]['valid_loss'] for i in range(len(test_samples))]
        test_dataset = tf.data.Dataset.from_tensor_slices({'vars':test_x,'valid_loss':test_y})
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'])
   
        
        return train_dataset, test_dataset, test_dataset
    
class DNNSumReduce(DNNWeightsLoader):
    def __init__(self, dataloader_args):
        super(DNNSumReduce, self).__init__(dataloader_args=dataloader_args)

    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path, name)
        yaml_feature_config = get_yml_content(yaml_path)
        info = self.get_info_inference(yaml_feature_config['num_of_students'],
                                            yaml_feature_config['sample_per_student'])
        
        feature_config = {
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'value', "length": 1, "dtype": tf.string},
            }

        return feature_config, info

    def _make_sumreduce_describs(self, feature_config=None):
        
        feature_describs = {}
        for feature, info in feature_config.items():
            if info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        feature_describs["vars_length"] = tf.io.FixedLenFeature([], tf.int64)
        return feature_describs

    def _load_tensor_from_tfrecord(self, filelist, feature_config):

        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        analyse_feature_describ = self._make_sumreduce_describs(
            feature_config = feature_config)

        def _parse_weights_function(example_proto):
            example = tf.io.parse_example(
                example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_weights_function,
                                                         num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset