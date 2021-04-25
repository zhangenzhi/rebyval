
import os, fnmatch
import tensorflow as tf
from tensorflow.io import gfile

from rebyval.dataloader.dataspec import BaseDataSpec
from rebyval.dataloader.utilis import glob_tfrecords

class BaseDataLoader:
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.dataspec = self._build_dataset_spec()

    def _build_dataset_spec(self):
        dataspec = BaseDataSpec(self.dataloader_args['dataspec'])
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
