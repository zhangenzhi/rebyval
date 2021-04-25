import os
import sys
import json
import tempfile
import string
import random
import ruamel.yaml as yaml
from colorama import Fore

from .constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO
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

def print_error(*content):
    '''Print error information to screen'''
    print(Fore.RED + ERROR_INFO + ' '.join([str(c)
                                            for c in content]) + Fore.RESET)


def print_green(*content):
    '''Print information to screen in green'''
    print(Fore.GREEN + ' '.join([str(c) for c in content]) + Fore.RESET)


def print_normal(*content):
    '''Print error information to screen'''
    print(NORMAL_INFO, *content)


def print_warning(*content):
    '''Print warning information to screen'''
    print(Fore.YELLOW + WARNING_INFO + ' '.join([str(c) for c in content]) +
          Fore.RESET)

def generate_temp_dir():
    '''generate a temp folder'''
    def generate_folder_name():
        return os.path.join(
            tempfile.gettempdir(), 'rebyval',
            ''.join(random.sample(string.ascii_letters + string.digits, 8)))

    temp_dir = generate_folder_name()
    while os.path.exists(temp_dir):
        temp_dir = generate_folder_name()
    os.makedirs(temp_dir)
    return temp_dir

def check_tensorboard_version():
    try:
        import tensorboard
        return tensorboard.__version__
    except:
        print_error('import tensorboard error!')
        exit(1)


if __name__ == '__main__':
    print_error("lalalal")
