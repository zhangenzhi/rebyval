import os
import sys
sys.path.append("./")
import tensorflow as tf

from rebyval.optimizer.scheduler.linear_scaling_with_warmup import *


def get_target_dataset(name=None):
    if name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        print("no such dataset")
        raise

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return train_dataset, test_dataset


def get_value_net():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='mse',
                  metrics=['mse'])
    return model


def get_dnn_target_net():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    lr_scheduler = LinearScalingWithWarmupSchedule(10, base_learning_rate=0.0004, warmup_steps=4000, gradual_steps=80000)
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_value_net():
    value_net = get_value_net()


def train_target_net():
    # train_dataset, test_dataset = get_target_dataset('cifar10')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_net = get_dnn_target_net()
    train_net.fit(x_train, y_train, batch_size= 32, epochs=10000, validation_data=(x_test,y_test))

    train_net.evaluate(x_test, y_test, batch_size= 32, verbose=2)


def save_train_net_vars(path, num, vars, val_loss):
    features = tf.train.Feature(feature={
        "label": tf.train.Feature(float32_list=tf.train.Float32List(value=[val_loss])),
        "target_weight": tf.train.Feature(float32_list=tf.train.Float32List(value=[vars]))
    })
    example = tf.train.Example(features=features)

    writer = tf.python_io.TFRecordWriter(os.path.join(path, num + ".tfrecord"))
    writer.write(example.SerializeToString())
    writer.close()


def main():
    for v_loop in range(100):
        for t_loop in range(500):
            train_target_net()
        train_value_net()


if __name__ == '__main__':
    train_target_net()
