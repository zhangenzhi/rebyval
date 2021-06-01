import io
import os
import time
import tarfile
import logging
from scipy import io as scipy_io

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from rebyval.model.resnet import ResNet50
from rebyval.dataloader.dataset_loader import ImageNetDataLoader
from rebyval.dataloader.utils import *


def get_conv_target_net():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.SGD(0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def load_ImageNet(ds_type, BASEDIR, batch_size):
    read_config = tfds.ReadConfig(num_parallel_calls_for_interleave_files=16,
                                  num_parallel_calls_for_decode=16)
    [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train', 'validation'],
                                             data_dir=BASEDIR, download=False, shuffle_files=True,
                                             read_config=read_config,
                                             batch_size=batch_size, as_supervised=False, with_info=True)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return [ds_train, ds_test], ds_info


if __name__ == '__main__':
    # input_dirs = "/home/work/dataset/ILSVRC2012/downloads/manual/train"
    # output_dirs = "/home/work/dataset/ILSVRC2012/downloads/manual/train_records"
    # metadata = convert_imagenet_to_tfrecords(input_dirs, output_dirs)

    dataloader_args = {'batch_size': 1024,
                       'datapath': "/home/work/dataset/ILSVRC2012/downloads/manual/train_records",
                       'sample_of_curves': 250}
    dataloader = ImageNetDataLoader(dataloader_args=dataloader_args)
    train_dataset, _, _ = dataloader.load_dataset()

    iter_train = iter(train_dataset)
    mean_t = tf.keras.metrics.Mean(name="test_avg_time")
    for _ in range(200):
        st = time.time()
        x = iter_train.get_next()
        et = time.time()
        mean_t(et-st)
        print("time cost:{} , avg time cost: {}".format(et - st, mean_t.result()))
