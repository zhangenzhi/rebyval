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


@tf.function(experimental_relax_shapes=True, experimental_compile=None)
def decode_image(image_raw, batch_size):
    decoded_image_batch = []
    for i in range(batch_size):
        decoded_image = tf.io.decode_image(image_raw[i], channels=3)
        resized_image = tf.image.resize(decoded_image, [256, 256])
        resized_image = tf.expand_dims(resized_image, axis=0)
        decoded_image_batch.append(resized_image)
    decoded_image = tf.concat(decoded_image_batch, axis=0)
    return decoded_image


if __name__ == '__main__':

    dataloader_args = {'batch_size': 256,
                       'datapath': "/home/work/dataset/ILSVRC2012/downloads/manual/train_records",
                       'sample_of_curves': 250}
    dataloader = ImageNetDataLoader(dataloader_args=dataloader_args)
    train_dataset, _, _ = dataloader.load_dataset()

    iter_train = iter(train_dataset)
    mean_t = tf.keras.metrics.Mean(name="test_avg_time")
    for i in range(200):
        st = time.time()
        x = iter_train.get_next()
        decoded_x = decode_image(x['image_raw'], batch_size=dataloader_args['batch_size'])
        et = time.time()
        if i != 0:
            mean_t(et - st)
        print("time cost:{} , avg time cost: {}".format(et - st, mean_t.result()))
