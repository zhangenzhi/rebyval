
import os
import random
import tensorflow as tf

from rebyval.dataloader.utils import glob_tfrecords
from rebyval.tools.utils import get_yml_content, print_green
from rebyval.dataloader.weights_loader import DNNWeightsLoader



class ChainLoader(DNNWeightsLoader):
    
    def __init__(self, dataloader_args):
        super(ChainLoader, self).__init__(dataloader_args=dataloader_args)
    
    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path, name)
        yaml_feature_config = get_yml_content(yaml_path)
        info = self.get_info_inference(yaml_feature_config['num_of_students'],
                                            yaml_feature_config['sample_per_student'])
        
        feature_config = {
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'previous_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'list', "length": yaml_feature_config['vars_length']['value'], "dtype": tf.string},
            }

        return feature_config, info
    