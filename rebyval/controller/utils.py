import os
from rebyval.controller.constants import default_values_dict

def check_args_from_yaml_content(yaml_content):
    checked_args = check_and_set_default_values(yaml_content)
    return checked_args


def check_and_set_default_values(experiment_config):
    return experiment_config