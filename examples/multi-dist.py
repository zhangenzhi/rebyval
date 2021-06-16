import tensorflow as tf
import os
import json
import mnist

import pdb
# pdb.set_trace()

f = open("./examples/experiment_configs/imagenet/tf_workers_config.json")
tf_config = json.load(f)
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
num_workers = len(tf_config['cluster']['worker'])

global_batch_size = 64 * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = mnist.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
