from distutils import filelist
from importlib.resources import path
import os
import sys
import json
import tempfile
import string
import random
import ruamel.yaml as yaml
from colorama import Fore

from rebyval.tools.constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO

def check_mkdir(path):
    if not os.path.exists(path=path):
        print_warning("no such path: {}, but we made.".format(path))
        os.makedirs(path)
        

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
        
def save_yaml_contents(file_path, contents):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(contents, file)
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

def print_red(*content):
    '''Print information to screen in red'''
    print(Fore.RED + ' '.join([str(c) for c in content]) + Fore.RESET)

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


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print_green('\t' * (indent) + key + ": ")
            print_dict(value, indent + 1)
        else:
            try:
                print('\t' * (indent) + str(key) + ": " + str(value))
            except:
                print_error("value can not print: ", type(value))
                raise


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


def get_cvr_sample_num(dir_list):
    total_num = 0
    for part_dir in dir_list:
        part_dir_up = os.path.split(part_dir)[0]
        num_file = os.path.join(part_dir_up, 'filter_cnt', 'part-00000')
        with open(num_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                num = (line.split('\t')[1]).split('\n')[0]
                total_num += int(num)
    return total_num


def auto_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_train_valid_test_dir_local(root_dir, date='20200721'):
    train_dirs = []
    valid_dirs = []
    test_dirs = []
    path = os.path.join(root_dir, date, 'online_sample', 'tfRecord')
    train_dirs.append(path)
    valid_dirs.append(path)
    test_dirs.append(path)

    return train_dirs, valid_dirs, test_dirs


def write_log(file, msg):
    with open(file, 'a+') as f:
        f.write(msg)
        f.write('\n')
 
def load_tensorboard_2_dataframe(path, to_csv=False):
    import tensorflow as tf
    filelists = os.path.lexists(path)
    for e in tf.compat.v1.train.summary_iterator(filelists):
        print(e)


if __name__ == '__main__':
    load_tensorboard_2_dataframe(path="./log/multi-p-v1/tensorboard/")
