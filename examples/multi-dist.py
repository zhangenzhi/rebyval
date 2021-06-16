import tensorflow as tf
import json

tf_config = json.loads("./experiment_configs/tf_workers_config.json")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
