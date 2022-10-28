import os

from tqdm import trange
from datetime import datetime
import tensorflow as tf

# import horovod.tensorflow as hvd

# dataloader
from rebyval.dataloader.dataset_factory import dataset_factory

# model
from rebyval.model.model_factory import model_factory

# others
from rebyval.plot.visualization import visualization
from rebyval.tools.utils import print_green, print_error, print_normal, check_mkdir, save_yaml_contents
from rebyval.dataloader.utils import glob_tfrecords


class Student(object):
    def __init__(self, student_args, supervisor=None, id=0, dist=False):
        self.args = student_args
        self.supervisor = supervisor
        self.id = id
        self.best_metrics = 0
        self.logdir = self._create_logdir()

    def _load_supervisor_model(self, supervisor_info, model_name="DNN_latest"):
        if self.supervisor == None:
            if supervisor_info==None:
                return None
            else:
                sp_model_logdir = os.path.join(supervisor_info["logdir"], model_name)
                supervisor = tf.keras.models.load_model(sp_model_logdir)
        else:
            supervisor = self.supervisor
        return supervisor

    def update_supervisor(self, inputs, labels):
        
        sp_opt = tf.keras.optimizers.SGD(0.01)
        sp_loss_fn = tf.keras.losses.MeanSquaredError()
        flat_vars = []
        for tensor in inputs:
            sum_reduce = tf.math.reduce_sum(tensor, axis=-1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(1, -1)))
        inputs = tf.concat(flat_vars, axis=1)
        
        with tf.GradientTape() as tape:
            predictions = self.supervisor(inputs)
            loss = sp_loss_fn(labels, predictions)
            gradients = tape.gradient(
                loss, self.supervisor.trainable_variables)
            sp_opt.apply_gradients(
                zip(gradients, self.supervisor.trainable_variables))
            
        for w in  self.supervisor.trainable_variables:
            w.assign(tf.clip_by_value(w, 0.0, 1.0))
            
        return loss

    def _build_enviroment(self, devices='0'):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_green("devices:", gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def _build_dataset(self):
        dataset_args = self.args['dataloader']
        dataloader = dataset_factory(dataset_args)

        train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()
        return train_dataset, valid_dataset, test_dataset, dataloader
    
    def _reset_dataset(self):
        self.train_dataset, self.valid_dataset, self.test_dataset, self.dataloader = self._build_dataset()
        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        return train_iter, valid_iter, test_iter
        

    def _build_model(self):
        model = model_factory(self.args['model'])
  
        # model restore
        if self.args['model'].get('restore_model'):
            self.model = self.model_restore(self.model)
        return model

    def _build_loss_fn(self):
        loss_fn = {}
        loss_fn = tf.keras.losses.get(self.args['loss']['name'])
        mt_loss_fn = tf.keras.metrics.Mean()
        mv_loss_fn = tf.keras.metrics.Mean()
        mtt_loss_fn = tf.keras.metrics.Mean()
        return loss_fn, mt_loss_fn, mv_loss_fn, mtt_loss_fn

    def _build_metrics(self):
        # metrics = {}
        metrics = self.args['metrics']
        train_metrics = tf.keras.metrics.get(metrics['name'])
        valid_metrics = tf.keras.metrics.get(metrics['name'])
        test_metrics = tf.keras.metrics.get(metrics['name'])
        return train_metrics, valid_metrics, test_metrics

    def _build_optimizer(self):
        optimizer_args = self.args['optimizer']
        optimizer = tf.keras.optimizers.get(optimizer_args['name'])
        optimizer.learning_rate = optimizer_args['learning_rate']
        self.base_lr = optimizer_args['learning_rate']

        return optimizer

    def _create_logdir(self):
        logdir = "tensorboard/" + "st-{}".format(self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args['log_path'], logdir)
        check_mkdir(logdir)
        return logdir
    
    def _build_logger(self):
        logger = tf.summary.create_file_writer(self.logdir)
        return logger

    def _build_writter(self):
        weight_dir = os.path.join(self.args['log_path'], "weight_space")
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
        save_path = os.path.join(self.logdir, '{}_{}'.format(type(self.model).__name__, name))

        save_msg = '\033[33m[Model Status]: Saving {} model in {:}.\033[0m'.format(name, save_path)
        print(save_msg)
        
        self.model.save(save_path, overwrite=True, save_format='tf')

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            self.train_metrics.update_state(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels):
        raise NotImplementedError("student need _rebyval_train_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        self.valid_metrics.update_state(labels, predictions)
        self.mv_loss_fn.update_state(loss)
        return loss

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        self.test_metrics.update_state(labels, predictions)
        self.mtt_loss_fn.update_state(loss)
        return loss

    def train(self, supervisor_info=None):
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        train_args = train_loop_args['train']
        valid_args = train_loop_args['valid']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        
        total_epochs = self.dataloader.info['epochs']  
        train_steps_per_epoch = self.dataloader.info['train_step']

        # load supervisor
        self.supervisor = self._load_supervisor_model(supervisor_info)

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(total_epochs, desc="Epochs") as e:
            for epoch in e:
                
                # lr decay
                if train_args["lr_decay"]:
                    if epoch == int(0.5*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print_green("Current decayed learning rate is {}".format(self.optimizer.learning_rate.numpy()))
                    elif epoch == int(0.75*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print_green("Current decayed learning rate is {}".format(self.optimizer.learning_rate.numpy()))

                # train
                with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    self.train_metrics.reset_states()
                    for train_step in t:
                        
                        # valid
                        if train_step % valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                for valid_step in v:
                                    v_data = valid_iter.get_next()
                                    valid_loss = self._valid_step(v_data['inputs'], v_data['labels'])
                                    v.set_postfix(sv_loss=valid_loss.numpy())
                                ev_loss = self.mv_loss_fn.result()
                                self.collect_test_metrics(current_state=self.model.trainable_variables,
                                                          metric=ev_loss,
                                                          format=valid_args['weight_space'])
                                
                                # get exp grad & online update supervisor
                                if train_args["ireval"] and self.supervisor != None:
                                    with tf.GradientTape() as tape_s:
                                        s_loss = self.weightspace_loss(self.model.trainable_variables, 
                                                                       format=valid_args['weight_space'])
                                    self.exp_grad = tape_s.gradient(s_loss, self.model.trainable_variables)
                                    self.update_supervisor(self.model.trainable_variables, ev_loss)
                        # train
                        if self.supervisor == None:
                            data = train_iter.get_next()
                            train_loss,_,  = self._train_step(data['inputs'], data['labels'])
                            t.set_postfix(st_loss=train_loss.numpy())
                        elif train_args["ireval"]:
                            data = train_iter.get_next()
                            v_data = valid_iter.get_next()
                            train_loss, exp_loss = self._ireval_train_step(data['inputs'], data['labels'], 
                                                                           format=valid_args['weight_space'])
                            with self.logger.as_default():
                                tf.summary.scalar("exp_loss", exp_loss, step=train_step + epoch*train_steps_per_epoch)
                            t.set_postfix(st_loss=train_loss.numpy(), pt_loss=exp_loss.numpy())
                        else:
                            data = train_iter.get_next()
                            train_loss, exp_loss = self._reval_train_step(data['inputs'], data['labels'])
                            with self.logger.as_default():
                                tf.summary.scalar("exp_loss", exp_loss, step=train_step + epoch*train_steps_per_epoch)
                            t.set_postfix(st_loss=train_loss.numpy(), pt_loss=exp_loss.numpy())
                                    
                    etr_loss = self.mt_loss_fn.result()
                    etr_metric = self.train_metrics.result()
                
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mtt_loss_fn.reset_states()
                    self.test_metrics.reset_states()
                    for test_step in t:
                        t_data = test_iter.get_next()
                        t_loss = self._test_step(t_data['inputs'], t_data['labels'])
                        t.set_postfix(test_loss=t_loss.numpy())
                    ete_loss = self.mtt_loss_fn.result()
                    ete_metric = self.test_metrics.result()
                    
                    # save best mdoel
                    if self.best_metrics < ete_metric and epoch%10==0:
                        self.model_save(name="best")
                        self.best_metrics = ete_metric

                    
                e.set_postfix(etr_loss=etr_loss.numpy(), etr_metric=etr_metric.numpy(), ete_loss=ete_loss.numpy(), 
                              ete_metric=ete_metric.numpy(), lr = self.optimizer.learning_rate.numpy())
                
                # train_iter, valid_iter, test_iter = self._reset_dataset()
                with self.logger.as_default():
                    tf.summary.scalar("etr_loss", etr_loss, step=epoch)
                    tf.summary.scalar("etr_metric", etr_metric, step=epoch)
                    tf.summary.scalar("ete_loss", ete_loss, step=epoch)
                    tf.summary.scalar("ete_metric", ete_metric, step=epoch)
        
        self.model.summary()
        self.model_save(name="finished")
        if train_loop_args["visual"]:
            visualization(self.model, 
                          train_iter.get_next(), test_iter.get_next(), 
                          step_size=1e-2, scale=100, 
                          save_to=self.logdir)

    def run(self, connect_queue=None, supervisor_info=None, devices='1'):

        # set enviroment
        self._build_enviroment(devices=devices)

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
            self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.loss_fn, self.mt_loss_fn, self.mv_loss_fn, self.mtt_loss_fn = self._build_loss_fn()
        self.train_metrics, self.valid_metrics, self.test_metrics = self._build_metrics()
        
        # build weights save writter
        self.logger = self._build_logger()
        self.writter, weight_dir = self._build_writter()

        self.train(supervisor_info=supervisor_info)

        self.writter.close()
        print('Finished training student {}'.format(self.id))

        if connect_queue != None:
            connect_queue.put(weight_dir)

        return weight_dir

    # weights space
    def collect_test_metrics(self, current_state=None, metric=None, format=None):
        self._write_trace_to_tfrecord(weights = current_state, valid_loss = metric, weight_space = format)

    def tensor_example(self, weight_loss):
        feature = {}
        configs = {}
        for feature_name, value in weight_loss.items():

            # weights config
            if isinstance(value, list):
                for i in range(len(value)):
                    bytes_v = tf.io.serialize_tensor(value[i]).numpy()
                    feature[feature_name + "_{}".format(i)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[bytes_v]))
                    configs[feature_name + "_{}".format(i)] = {
                        'type': 'bytes', 'shape': value[i].shape.as_list()}

                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(value)])
                )
                configs[feature_name +
                        "_length"] = {'type': 'int64', 'shape': [1], 'value': len(value)}

            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value))
                configs[feature_name] = {'type': 'float32', 'shape': [1]}
        return tf.train.Example(features=tf.train.Features(feature=feature)), configs
    
    def flatten_example(self, weight_loss):
        feature = {}
        configs = {}
        for feature_name, value in weight_loss.items():
            if isinstance(value, list):
                model_vars = []
                for tensor in value:
                    compressed_tensor = tensor
                    model_vars.append(tf.reshape(
                        compressed_tensor, shape=(-1)))
                model_vars = tf.concat(model_vars, axis=0)
                bytes_v = tf.io.serialize_tensor(model_vars).numpy()
                #vars
                feature[feature_name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes_v]))
                configs[feature_name] = {'type': 'bytes', 'shape': model_vars.shape.as_list()}
                #vars_length
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[1])
                )
                configs[feature_name +
                        "_length"] = {'type': 'int64', 'shape': [1], 'value': 1}

            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value))
                configs[feature_name] = {'type': 'float32', 'shape': [1]}
        return tf.train.Example(features=tf.train.Features(feature=feature)), configs

    def sum_reduce_example(self, weight_loss):
        feature = {}
        configs = {}
        for feature_name, value in weight_loss.items():
            if isinstance(value, list):
                model_vars = []
                for tensor in value:
                    axis = tensor.shape.rank - 1
                    compressed_tensor = tf.math.reduce_sum(
                        tensor, axis=axis, keepdims=True)
                    model_vars.append(tf.reshape(
                        compressed_tensor, shape=(-1)))
                model_vars = tf.concat(model_vars, axis=0)
                bytes_v = tf.io.serialize_tensor(model_vars).numpy()
                #vars
                feature[feature_name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes_v]))
                configs[feature_name] = {'type': 'bytes', 'shape': model_vars.shape.as_list()}
                #vars_length
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[1])
                )
                configs[feature_name +
                        "_length"] = {'type': 'int64', 'shape': [1], 'value': 1}

            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value))
                configs[feature_name] = {'type': 'float32', 'shape': [1]}
        return tf.train.Example(features=tf.train.Features(feature=feature)), configs
    


    def _write_trace_to_tfrecord(self, weights, valid_loss, weight_space=None):

        weight_loss = {'vars': weights, 'valid_loss': valid_loss}

        if weight_space == 'flatten':
            example, configs = self.flatten_example(weight_loss)
        elif weight_space == 'sum_reduce':
            example, configs = self.sum_reduce_example(weight_loss)
        else:
            example, configs = self.tensor_example(weight_loss)

        weight_dir = os.path.join(self.args['log_path'], 'weight_space')
        config_path = os.path.join(weight_dir, 'feature_configs.yaml')

        configs['num_of_students'] = len(glob_tfrecords(
            weight_dir, glob_pattern='*.tfrecords'))
        configs['sample_per_student'] = int(
            self.dataloader.info['train_step'] / self.args['train_loop']['valid']['valid_gap'] + 1) * self.dataloader.info['epochs']
        configs['total_samples'] = configs['sample_per_student'] * \
            configs['num_of_students']
        save_yaml_contents(contents=configs, file_path=config_path)

        self.writter.write(example.SerializeToString())
            