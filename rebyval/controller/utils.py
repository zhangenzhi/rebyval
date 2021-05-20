import os
from rebyval.controller.constants import default_values_dict


def check_args_from_yaml_content(yaml_content):
    checked_args = check_and_set_default_values(yaml_content)
    return checked_args


def check_and_set_default_values(experiment_config):
    return experiment_config


def valid_weights_pool(surrogate_args):
    dataloader_args = surrogate_args['dataloader']

    if isinstance(surrogate_args['dataloader']['datapath'], str):
        weights_pool_path = os.path.join(dataloader_args['datapath'],dataloader_args['format'])
        subfiles = os.listdir(weights_pool_path)
        if subfiles == []:
            raise ("No weights in the pool.")
        new_datapath = [os.path.join(weights_pool_path, f) for f in subfiles]
        surrogate_args['dataloader']['datapath'] = new_datapath

    elif isinstance(surrogate_args['dataloader']['datapath'], list):
        weights_pool_path = os.path.join(surrogate_args['dataloader']['datapath'], '..')
        subfiles = os.listdir(weights_pool_path)
        if subfiles == []:
            raise ("No weights in the pool.")
        new_datapath = [os.path.join(weights_pool_path, f) for f in subfiles]
        surrogate_args['dataloader']['datapath'] = new_datapath
    else:
        raise ("No such type of datapath")
