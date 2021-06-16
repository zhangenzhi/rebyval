import tensorflow as tf
import json
import pdb

pdb.set_trace()

tf_config = json.loads("./experiment_configs/tf_workers_config.json")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
