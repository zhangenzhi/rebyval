import os
from rebyval.tools.utils import *
from rebyval.controller.constants import default_parameters


def check_args_from_input_config(input_config):
    default_configs = get_yml_content("./rebyval/controller/default_parameters.yaml")
    merged_configs = check_and_merge(input_config, default_configs)
    print_dict(merged_configs)
    return merged_configs

def check_and_merge(input_config, default_configs, indent=0):
    for key, value in default_configs.items():
        if key not in input_config:
            input_config[key] = value
            if isinstance(value, dict):
                    check_and_merge(input_config[key], value, indent + 1)
        else:
            if isinstance(value, dict):
                    check_and_merge(input_config[key], value, indent + 1)
    return input_config

def valid_weights_pool(dataloader_args):
    if isinstance(dataloader_args['datapath'], str):
        weights_pool_path = os.path.join(dataloader_args['datapath'], dataloader_args['format'])
        subfiles = os.listdir(weights_pool_path)
        if subfiles == []:
            raise ("No weights in the pool.")
        new_datapath = [os.path.join(weights_pool_path, f) for f in subfiles]
        dataloader_args['datapath'] = new_datapath

    elif isinstance(dataloader_args['datapath'], list):
        weights_pool_path = os.path.join(dataloader_args['datapath'][0], '..')
        subfiles = os.listdir(weights_pool_path)
        if subfiles == []:
            raise ("No weights in the pool.")
        new_datapath = [os.path.join(weights_pool_path, f) for f in subfiles]
        dataloader_args['datapath'] = new_datapath
    else:
        raise ("No such type of data.")
