import tensorflow as tf
import os
import json
import pdb
pdb.set_trace()
f = open("./examples/experiment_configs/imagenet/tf_workers_config.json")
os.environ['TF_CONFIG'] = json.load(f)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
