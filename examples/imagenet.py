import io
import os
import time
import tarfile
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from rebyval.model.resnet import ResNet50


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


if __name__ == '__main__':
    # imagenet_ds = ImageNet2012()
    dataset_name = "imagenet2012"
    manual_dataset_dir = "/home/work/dataset/ILSVRC2012"


    # builder = tfds.builder(dataset_name, data_dir=manual_dataset_dir)
    # tfds.list_builders()
    # tfds.load('mnist')
    def load_ImageNet(ds_type, BASEDIR, batch_size):
        read_config = tfds.ReadConfig()
        [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train', 'validation'],
                                                 data_dir=BASEDIR, download=True, shuffle_files=True,
                                                 read_config=read_config,
                                                 batch_size=batch_size, as_supervised=True, with_info=True)
        # ds_train = ds_train.interleave(lambda x,y: tf.data.TFRecordDataset(x),
        #                                block_length=256,
        #                                cycle_length=16,
        #                                num_parallel_calls=16,
        #                                deterministic=False)

        resize_and_rescale = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(256, 256),
                                                  layers.experimental.preprocessing.Rescaling(1. / 255.)])

        ds_train = ds_train.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        # ds_train = ds_train.cache()
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        return [ds_train, ds_test], ds_info


    [ds_train, ds_test], ds_info = load_ImageNet(dataset_name, BASEDIR=manual_dataset_dir, batch_size=64)

    train_iter = iter(ds_train)
    mean = tf.keras.metrics.Mean(name="avg_time")
    for _ in range(200):
        st = time.time()
        x = train_iter.get_next()
        et = time.time()
        mean(et-st)
        print("cost time: {},avg time: {}".format(et-st,mean.result()))
    # model = get_conv_target_net()
    # model.fit(ds_train, epochs=1, validation_data=ds_test)

    # ResNet50
    # net = ResNet50()
    # train_iter = iter(ds_train)
    # for _ in range(10):
    #     import pdb
    #     pdb.set_trace()
    #     x = train_iter.get_next()
    #
    # net(x)
