
import os
from datetime import datetime
import tensorflow as tf

# dataloader
from rebyval.dataloader.dataset_loader import Cifar10DataLoader

# model
from rebyval.model.dnn import DNN
from rebyval.model.cnn import CNN

# others
from rebyval.train.utils import ForkedPdb
from rebyval.tools.utils import print_green, print_error, print_normal, check_mkdir, save_yaml_contents
from rebyval.dataloader.utils import glob_tfrecords


class Student:
    def __init__(self, student_args, supervisor = None, supervisor_vars = None , id = 0):
        self.args = student_args
        self.supervisor = supervisor
        self.supervisor_vars  = supervisor_vars
        self.id = id
        
    def _build_supervisor_from_vars(self):
        model = None
        if self.supervisor_vars != None:
            #TODO: need model registry
            model = DNN(units=[128, 64, 32, 16, 1],
                    activations=['relu','relu', 'relu', 'relu', 'softplus'],
                    use_bn=False,
                    initial_value=self.supervisor_vars,
                    seed=None)
            
        return model
    
    def update_supervisor(self, inputs, labels):
        supervisor_opt = tf.keras.optimizers.SGD(0.01)
        supervisor_loss_fn = tf.keras.losses.mae
        flat_vars = []
        for tensor in inputs:
            sum_reduce = tf.math.reduce_sum(tensor, axis= -1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(1, -1)))
        inputs = tf.concat(flat_vars, axis=1)
        
        with tf.GradientTape() as tape:
            predictions = self.supervisor(inputs, training=True)
            predictions = tf.squeeze(predictions)
            loss = supervisor_loss_fn(labels, predictions)

            gradients = tape.gradient(loss, self.supervisor.trainable_variables)

            supervisor_opt.apply_gradients(
                zip(gradients, self.supervisor.trainable_variables))
        print(loss)
        

    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_green("devices:", gpus)
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
        model = DNN(units=[64,32,16,10],
                    activations=['relu', 'relu', 'relu', 'softmax'])
        # model = CNN()
        # model restore
        if self.args['model'].get('restore_model'):
            self.model = self.model_restore(self.model)
        return model

    def _build_loss_fn(self):
        loss_fn = {}
        loss_fn = tf.keras.losses.get(self.args['loss']['name'])
        mt_loss_fn = tf.keras.metrics.Mean()
        mv_loss_fn = tf.keras.metrics.Mean()
        return loss_fn, mt_loss_fn, mv_loss_fn

    def _build_metrics(self):
        metrics = {}
        metrics = self.args['metrics']
        metrics = tf.keras.metrics.get(metrics['name'])
        return metrics

    def _build_optimizer(self):
        optimizer_args = self.args['optimizer']
        optimizer = tf.keras.optimizers.get(optimizer_args['name'])
        optimizer.learning_rate = optimizer_args['learning_rate']
        return optimizer
    
    def _build_logger(self):
        logdir = "tensorboard/" + "student-{}-".format(self.id) + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args['log_path'], logdir)
        check_mkdir(logdir)
        logger = tf.summary.create_file_writer(logdir)
        return logger
    
    def _build_writter(self):
        weight_dir = os.path.join(self.args['log_path'],"weight_space")
        check_mkdir(weight_dir)
        weight_trace = os.path.join(weight_dir, '{}.tfrecords'.format(self.id))
        writter = tf.io.TFRecordWriter(weight_trace)
        return writter, weight_trace

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
        train_loop_args = self.args['train_loop']
        train_args = train_loop_args['train']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        # metrics reset
        metric_name = self.args['metrics']['name']
        self.metrics[metric_name].reset_states()
        
        raise NotImplementedError("need train, valid, test logic.")

    def run(self, new_student=None, supervisor_vars = None):

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
        self.loss_fn, self.mt_loss_fn, self.mv_loss_fn = self._build_loss_fn()
        self.metrics = self._build_metrics()
        
        # build weights save writter
        self.logger = self._build_logger()
        self.writter, weight_dir = self._build_writter()

        # # train
        # import pdb
        # pdb.set_trace()
      
        # self.supervisor = self._build_supervisor_from_vars()
        self.train(supervisor_vars = supervisor_vars)
        
        self.writter.close()
        print('Finished training student {}'.format(self.id))
        
        new_student.put(weight_dir)
        
        return weight_dir
        

    # weights space
    def _during_train_vars_tensor_example(self, weight_loss):
        feature = {}
        configs = {}
        for feature_name, value in weight_loss.items():
            
            # weights config
            if isinstance(value, list):
                for i in range(len(value)):
                    bytes_v = tf.io.serialize_tensor(value[i]).numpy() 
                    feature[feature_name + "_{}".format(i)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[bytes_v]))
                    configs[feature_name + "_{}".format(i)]= {'type':'bytes','shape':value[i].shape.as_list()}
                    
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(value)])
                )
                configs[feature_name + "_length"]= {'type':'int64','shape':[1],'value':len(value)}
                
            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
                configs[feature_name]= {'type':'float32', 'shape':[1]}
        return tf.train.Example(features=tf.train.Features(feature=feature)), configs

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

    def _write_trace_to_tfrecord(self, weights, valid_loss, weight_space=None):
        
        weight_loss = {'vars': weights, 'valid_loss': valid_loss}
        
        if weight_space['format'] == 'tensor':
            example,configs = self._during_train_vars_tensor_example(weight_loss)
        elif weight_space['format'] == 'tensor_sum_reduce':
            example = self._during_train_vars_tensor_sum_reduce_example()
        elif weight_space['format'] == 'tensor_sum_reduce_l2':
            example = self._during_train_tensor_sum_reduce_with_l2_example()
        else:
            example = self._during_train_vars_numpy_example()
 
        weight_dir =  os.path.join(self.args['log_path'], 'weight_space')
        config_path = os.path.join(weight_dir, 'feature_configs.yaml')
        
        configs['num_of_students'] = len(glob_tfrecords(weight_dir, glob_pattern='*.tfrecords'))
        configs['sample_per_student'] = int(self.dataloader.info['train_step']  / self.args['train_loop']['valid']['valid_gap'] + 1) * self.dataloader.info['epochs']
        configs['total_samples'] = configs['sample_per_student']  * configs['num_of_students']
        save_yaml_contents(contents=configs, file_path=config_path)
        
        self.writter.write(example.SerializeToString())
        


