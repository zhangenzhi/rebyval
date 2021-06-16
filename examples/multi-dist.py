import tensorflow as tf
import json
import pdb

pdb.set_trace()
f = open("./experiment_configs/tf_workers_config.json")
tf_config = json.load(f)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
