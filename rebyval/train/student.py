import os
import tensorflow as tf

# dataloader
from rebyval.dataloader.dataset_loader import Cifar10DataLoader, ImageNetDataLoader
from rebyval.dataloader.weights_loader import DnnWeightsLoader

# model
from rebyval.model.dnn import DenseNeuralNetwork
from rebyval.model.resnet import ResNet50

# others
from rebyval.train.utils import get_scheduler, prepare_dirs
from rebyval.tools.utils import calculate_auc, write_log, print_green, print_error, print_normal

class Student:
    def __init__(self, student_args, supervisor = None):
        self.args = student_args
        self.supervisor = supervisor

    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_green(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def _build_dataset(self):
        #TODO: need dataloader registry
        dataset_args = self.args['dataloader']
        dataloader = Cifar10DataLoader(dataset_args)
   
        train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()
        return train_dataset, valid_dataset, test_dataset, dataloader

    def _build_model(self):
        #TODO: need model registry
        model_args = self.args['model']
        model = DenseNeuralNetwork()
        # model restore
        if self.args['model'].get('restore_model'):
            self.model = self.model_restore(self.model)
        return model

    def _build_loss_fn(self):
        loss_fn = {}
        loss_fn = tf.keras.losses.get(self.args['loss']['name'])
        return loss_fn

    def _build_metrics(self):
        metrics = {}
        metrics_name = self.args['metrics']['name']
        metrics[metrics_name] = tf.keras.metrics.get(metrics_name)
        return metrics

    def _build_optimizer(self):
        optimizer_args = self.args['optimizer']
        optimizer = tf.keras.optimizers.get(optimizer_args['name'])
        return optimizer
    
    def _build_weights_writer(self):
        # weights writer
        filepath = self.valid_args['weightspace_path']
        record_file = os.path.join(filepath, '{}.tfrecords'.format(0))
        self.writer = tf.io.TFRecordWriter(record_file)

    def model_restore(self, model):
        model_args = self.args['model']
        model_path = model_args['restore_from']
        model.load_weights(model_path)
        print_normal("restore from {} success!".format(model_path))
        return model

    def model_save(self, name):
        save_path = os.path.join(self.valid_args['model_dir'], self.valid_args['save_model']['save_in'])
        save_path = os.path.join(save_path, 'model_{}'.format(name))

        save_msg = '\033[33m[Model Status]: Saving {} model at step:{:08d} in {:}.\033[0m'.format(
            name, self.global_step, save_path)
        print(save_msg)
    
        self.model.save_weights(save_path, overwrite=True, save_format='tf')

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        raise NotImplementedError("student need _train_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels):
        raise NotImplementedError("student need _rebyval_train_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        raise NotImplementedError("student need _valid_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        raise NotImplementedError("student need _test_step func.")

        
    def train(self):

        # parse train loop control args
        train_loop_control_args = self.args['train_loop_control']
        train_args = train_loop_control_args['train']

        # dataset train, valid, test
        epoch = 0
        step = 0
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        # metrics reset
        metric_name = self.args['metrics']['name']
        self.metrics[metric_name].reset_states()
        
        raise NotImplementedError("need train, valid, test logic.")

    def run(self):

        # set enviroment
        self._build_enviroment()

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
        self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.loss_fn = self._build_loss_fn()
        self.metrics = self._build_metrics()

        # train
        self.train()

        # Analyse
    def _during_train_vars_tensor_example(self):
        feature = {}
        for feature_name, value in self.during_value_dict.items():
            if isinstance(value, list):
                value = [tf.io.serialize_tensor(v).numpy() for v in value]
                v_len = len(value)
                for i in range(v_len):
                    feature[feature_name + "_{}".format(i)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value[i]]))
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[v_len])
                )
            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _during_train_vars_tensor_sum_reduce_example(self):
        feature = {}
        for feature_name, value in self.during_value_dict.items():
            if isinstance(value, list):
                model_vars = []
                for tensor in value:
                    axis = tensor.shape.rank - 1
                    compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                    model_vars.append(tf.reshape(compressed_tensor, shape=(-1)))
                model_vars = tf.concat(model_vars, axis=0)
                value = tf.io.serialize_tensor(model_vars).numpy()
                feature[feature_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _during_train_tensor_sum_reduce_with_l2_example(self):
        feature = {}
        for feature_name, value in self.during_value_dict.items():
            if isinstance(value, list):
                model_vars = []

                # drop weights of first layer
                value = value[2:]

                for tensor in value:
                    axis = tensor.shape.rank - 1
                    compressed_tensor = tf.math.reduce_sum(tensor, axis=axis, keepdims=True)
                    model_vars.append(tf.reshape(compressed_tensor, shape=(-1)))
                model_vars = tf.concat(model_vars, axis=0)
                value = tf.io.serialize_tensor(model_vars).numpy()
                feature[feature_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _during_train_vars_numpy_example(self):
        feature = {}
        for feature_name, value in self.during_value_dict.items():
            if isinstance(value, list):
                value = [v.numpy() for v in value]
                v_len = len(value)
                for i in range(v_len):
                    feature[feature_name + "_{}".format(i)] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[value[i]]))
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[v_len])
                )
            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_analyse_to_tfrecord(self):

        if self.valid_args['analyse']['format'] == 'tensor':
            example = self._during_train_vars_tensor_example()
        elif self.valid_args['analyse']['format'] == 'tensor_sum_reduce':
            example = self._during_train_vars_tensor_sum_reduce_example()
        elif self.valid_args['analyse']['format'] == 'tensor_sum_reduce_l2':
            example = self._during_train_tensor_sum_reduce_with_l2_example()
        else:
            example = self._during_train_vars_numpy_example()

        self.writer.write(example.SerializeToString())

