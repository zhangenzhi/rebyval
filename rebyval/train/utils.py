import os

def prepare_dirs(valid_args):

    if valid_args.get('log_path'):
        valid_args['log_file'] = os.path.join(valid_args['log_path'], 'log_file.txt')
        valid_args['model_dir'] = os.path.join(valid_args['log_path'], 'models')
        valid_args['tensorboard_dir'] = os.path.join(valid_args['log_path'], 'tensorboard')
        mkdirs(valid_args['model_dir'])
        mkdirs(valid_args['tensorboard_dir'])

        # weights pool
        if valid_args['analyse']:
            valid_args['analyse_dir'] = os.path.join(valid_args['log_path'], 'analyse')
            if not os.path.isdir(valid_args['analyse_dir']):
                mkdirs(valid_args['analyse_dir'])
            target_model_version = len(os.listdir(valid_args['analyse_dir']))
            valid_args['analyse_dir'] = os.path.join(valid_args['analyse_dir'],str(target_model_version))
            mkdirs(valid_args['analyse_dir'])


def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_scheduler(name):
    return name