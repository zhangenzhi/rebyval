import os
from datetime import datetime
import tensorflow as tf

from rebyval.dataloader.dataset_factory import dataset_factory

# model
from rebyval.model.model_factory import model_factory

# others
from rebyval.tools.utils import print_green, print_error, print_normal, check_mkdir
from rebyval.dataloader.utils import glob_tfrecords

class Supervisor(object):
    def __init__(self, supervisor_args, logger = None, id = 0):
        self.args = supervisor_args
        self.logger = logger
        self.id = id
        self.logdir = self._create_logdir()
    
    def __call__(self, weights):
        pass
    
    def _build_enviroment(self, devices='0'):

        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_green(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    def _build_model(self):
        model = model_factory(self.args['model'])

        # model restore
        if self.args['model'].get('restore_model'):
            self.model = self.model_restore(self.model)
        return model

    def _build_dataset(self, new_students = []):
        dataset_args = self.args['dataloader']
        
        datadir = "weight_space"
        dataset_args['path'] = os.path.join(self.args['log_path'], datadir)
        dataloader = dataset_factory(dataset_args)
        print_green("-"*10+"build_dataset"+"-"*10)
        train_dataset, valid_dataset, test_dataset = dataloader.load_dataset(new_students = new_students)
        return train_dataset, valid_dataset, test_dataset, dataloader

    def _build_loss_fn(self):
        loss_fn = {}
        loss_fn = tf.keras.losses.get(self.args['loss']['name'])
        mloss_fn = tf.keras.metrics.Mean()
        return loss_fn, mloss_fn
    
    def _build_metrics(self):
        metrics = {}
        metrics_name = self.args['metrics']['name']
        metrics[metrics_name] = tf.keras.metrics.get(metrics_name)
        return metrics
 
    def _build_optimizer(self):
        optimizer_args = self.args['optimizer']
        optimizer = tf.keras.optimizers.get(optimizer_args['name'])
        optimizer.learning_rate = optimizer_args['learning_rate']
        
        return optimizer
    
    def _create_logdir(self):
        logdir = "tensorboard/" + "sp-{}".format(self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args['log_path'], logdir)
        check_mkdir(logdir)
        return logdir
    
    def _build_logger(self):
        logger = tf.summary.create_file_writer(self.logdir)
        return logger

    def model_restore(self, name):
        path = os.path.join(self.logdir,'{}_{}'.format(type(self.model).__name__, name))
        model = tf.keras.models.load_model(path)
        print_normal("restore from {} success!".format(path))
        return model

    def model_save(self, name):
        save_path = os.path.join(self.logdir, '{}_{}'.format(type(self.model).__name__, name))
        save_msg = '\033[33m[Model Status]: Saving {} model in {:}.\033[0m'.format(name, save_path)
        print(save_msg)
        self.model.save(save_path)

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        raise NotImplementedError("supervisor need _train_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        raise NotImplementedError("supervisor need _valid_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        raise NotImplementedError("supervisor need _test_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _parse_tensor(self, x):
        parsed_tensors = {}
        for feat, tensor in x.items():
            batch_serilized_tensor = []
            for i in range(256):
                batch_serilized_tensor.append(tf.io.parse_tensor(tensor[i], tf.float32))
            parsed_tensors = tf.concat(batch_serilized_tensor, axis=0)
        return parsed_tensors
    
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

    def run(self, keep_train=False, new_students=[]):
        
        self.id += 1
        
        if keep_train:
            # prepare dataset
            print_green("-"*10+"run_keep"+"-"*10)
            self.new_students = new_students
            self.train_dataset, self.valid_dataset, self.test_dataset \
                = self.dataloader.load_dataset(new_students = new_students)
            
            # train
            self.train()
            
        else:
            # set enviroment
            print_green("-"*10+"run_init"+"-"*10)
            self._build_enviroment()

            # prepare dataset
            self.train_dataset, self.valid_dataset, self.test_dataset, \
            self.dataloader = self._build_dataset()

            # build optimizer
            self.optimizer = self._build_optimizer()

            # build model
            self.model = self._build_model()

            # build losses and metrics
            self.loss_fn, self.mloss_fn = self._build_loss_fn()
            
            # build weights and writter
            self.logger = self._build_logger()

            # train
            self.train()

