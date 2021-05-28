import io
import os
import tarfile
import logging

import tensorflow as tf
import tensorflow_datasets as tfds



if __name__ == '__main__':
    # imagenet_ds = ImageNet2012()
    dataset_name = "imagenet2012"
    manual_dataset_dir = "/home/work/dataset/ILSVRC2012"
    # builder = tfds.builder(dataset_name, data_dir=manual_dataset_dir)
    # tfds.list_builders()
    # tfds.load('mnist')
    def load_ImageNet(ds_type, BASEDIR, batch_size):
        [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train','validation'],
                                                 data_dir=BASEDIR, download=True, shuffle_files=True,
                                                 as_supervised=True, with_info=True)

        # ds_train = prepare_training(ds_train, batch_size)
        # ds_test = prepare_test(ds_test, batch_size)
        return [ds_train, ds_test], ds_info

    [ds_train, ds_test], ds_info = load_ImageNet(dataset_name,BASEDIR=manual_dataset_dir,batch_size=1024)
    # import pdb
    # pdb.set_trace()
    # ds_train.take(10)
