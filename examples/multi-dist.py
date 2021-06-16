import tensorflow as tf
import os
import json
import pdb
pdb.set_trace()
f = open("./examples/experiment_configs/imagenet/tf_workers_config.json")
tf_config = json.load(f)
os.environ['TF_CONFIG'] = json.dumps(tf_config)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
