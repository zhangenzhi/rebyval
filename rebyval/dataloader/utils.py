import fnmatch, os
from tensorflow.io import gfile

def glob_tfrecords(input_dirs, glob_pattern="example", recursively=False):
    file_path_list = []
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    for root_path in input_dirs:
        assert gfile.exists(root_path), "{} does not exist.".format(root_path)
        if not gfile.isdir(root_path):
            file_path_list.append(root_path)
            continue
        if not recursively:
            for filename in gfile.listdir(root_path):
                if fnmatch.fnmatch(filename, glob_pattern):
                    file_path_list.append(os.path.join(root_path, filename))
        else:
            for dir_path, _, filename_list in gfile.walk(root_path):
                for filename in filename_list:
                    if fnmatch.fnmatch(filename, glob_pattern):
                        file_path_list.append(os.path.join(dir_path, filename))
    return file_path_list