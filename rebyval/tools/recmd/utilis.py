# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import tempfile
import socket
import string
import random
import ruamel.yaml as yaml
import psutil
from colorama import Fore

from ..constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO


def get_format_time(cost_time):
    minute, second = divmod(cost_time, 60)
    hour, minute = divmod(minute, 60)
    formated_time = '{:d}h：{:d}m：{:.2f}s'.format(int(hour), int(minute),
                                                 second)
    return formated_time


def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.scanner.ScannerError as err:
        print_error('yaml file format error!')
        print_error(err)
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)


def get_json_content(file_path):
    '''Load json file content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print_error('json file format error!')
        print_error(err)
        return None



