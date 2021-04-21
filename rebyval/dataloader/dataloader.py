
import os, fnmatch
import tensorflow as tf
from tensorflow.io import gfile


class DataSpec:
    def __init__(self, dataspec_args):
        pass


class DataLoader:
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.dataspec = self._build_dataset_spec()

    def _build_dataset_spec(self):
        dataspec = DataSpec(self.dataloader_args['dataspec'])
        return dataspec

    def process_example(self, example):
        pass

    def load_dataset(self,
                     input_dirs,
                     glob_pattern,
                     glob_recursively=False):

        example_spec = self.dataspec

        file_list = glob_tfrecords(input_dirs,
                                   glob_pattern,
                                   recursively=glob_recursively)
        assert len(file_list) > 0, "No data files found."



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
