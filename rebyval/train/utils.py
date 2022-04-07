import os
import sys
import pdb
from rebyval.tools.utils import print_warning

def prepare_dirs(valid_args):
    if valid_args.get('log_path'):
        valid_args['log_file'] = os.path.join(valid_args['log_path'], 'log_file.txt')
        valid_args['model_dir'] = os.path.join(valid_args['log_path'], 'models')
        valid_args['tensorboard_dir'] = os.path.join(valid_args['log_path'], 'tensorboard')
        mkdirs(valid_args['model_dir'])
        mkdirs(valid_args['tensorboard_dir'])

        # weights pool
        if valid_args.get('analyse'):
            valid_args['analyse_dir'] = os.path.join(valid_args['log_path'],
                                                     'analyse/{}'.format(valid_args['analyse']['format']))
            if not os.path.isdir(valid_args['analyse_dir']):
                mkdirs(valid_args['analyse_dir'])
            target_model_version = len(os.listdir(valid_args['analyse_dir']))
            valid_args['analyse_dir'] = os.path.join(valid_args['analyse_dir'], str(target_model_version))
            valid_args['log_file'] = os.path.join(valid_args['analyse_dir'], 'log_file.txt')
            mkdirs(valid_args['analyse_dir'])

def check_mkdir(path):
    if not os.path.exists(path=path):
        print_warning("no such path: {}, but we made.".format(path))
        os.makedirs(path)
        
def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_scheduler(name):
    return name

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
