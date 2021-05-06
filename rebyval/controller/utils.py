import os
from rebyval.controller.constants import default_values_dict

def check_args_from_yaml_content(yaml_content):
    checked_args = check_and_set_default_values(yaml_content)
    return checked_args


def check_and_set_default_values(experiment_config):
    return experiment_config

def valid_weights_pool(surrogate_args):

    dataloader_args = surrogate_args['dataloader']
    weights_pool_path = dataloader_args['datapath']
    subfiles = os.listdir(weights_pool_path)
    if subfiles == []:
        print("No weights in the pool.")
        raise
    new_datapath = [os.path.join(weights_pool_path,f) for f in subfiles]
    surrogate_args['dataloader']['datapath'] = new_datapath