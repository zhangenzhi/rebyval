import tensorflow as tf
import json
import pdb

pdb.set_trace()
f = open("./imagenet/tf_workers_config.json")
tf_config = json.load(f)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
