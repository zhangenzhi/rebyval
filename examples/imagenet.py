import io
import os
import tarfile
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
from rebyval.model.resnet import ResNet50

def get_conv_target_net():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(None, None, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)
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
        [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train','validation'],
                                                 data_dir=BASEDIR, download=True, shuffle_files=True,
                                                 batch_size = batch_size, as_supervised=True, with_info=True)

        # ds_train = prepare_training(ds_train, batch_size)
        # ds_test = prepare_test(ds_test, batch_size)
        def normaliz_img(image,label):
            return tf.cast(image,tf.float32)/255.,label
        ds_train = ds_train.map(normaliz_img,num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(normaliz_img, num_parallel_calls=tf.data.AUTOTUNE)
        return [ds_train, ds_test], ds_info

    [ds_train, ds_test], ds_info = load_ImageNet(dataset_name,BASEDIR=manual_dataset_dir,batch_size=32)

    # ds_train.take(10)


    model = get_conv_target_net()
    model.fit(ds_train, epochs=1,validation_data=ds_test)

    # ResNet50
    # net = ResNet50()
    # train_iter = iter(ds_train)
    # for _ in range(10):
    #     import pdb
    #     pdb.set_trace()
    #     x = train_iter.get_next()
    #
    # net(x)
