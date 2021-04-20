import os
import tensorflow as tf


def get_target_dataset(name=None):
    if name == 'cifar10':
        target_dataset = tf.keras.datasets.cifar10
    else:
        print("no such dataset")
        raise
    return target_dataset


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


def get_target_net():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_value_net():
    value_net = get_value_net()



def train_target_net():
    dataset = get_target_dataset('cifar10')
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    train_net = get_target_net()
    train_net.fit(x_train, y_train, epochs=5)

    train_net.evaluate(x_test, y_test, verbose=2)

def save_train_net_vars(path, num, vars, val_loss):
    features = tf.train.Feature(feature={
        "label":tf.train.Feature(float32_list=tf.train.Float32List(value=[val_loss])),
        "target_weight":tf.train.Feature(float32_list=tf.train.Float32List(value=[vars]))
    })
    example = tf.train.Example(features=features)

    writer = tf.python_io.TFRecordWriter(os.path.join(path, num+".tfrecord"))
    writer.write(example.SerializeToString())
    writer.close()




def main():
    for v_loop in range(100):
        for t_loop in range(500):
            train_target_net()
        train_value_net()


if __name__ == '__main__':
    train_target_net()
