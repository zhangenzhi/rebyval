import os
from rebyval.controller.constants import default_values_dict

def check_args_from_yaml_content(yaml_content):
    checked_args = yaml_content
    return checked_args


def expand_path(experiment_config, key):
    '''Change '~' to user home directory'''
    if experiment_config.get(key):
        experiment_config[key] = os.path.expanduser(experiment_config[key])


def parse_path(experiment_config):
    pass


def set_default_values(experiment_config):
    pass
