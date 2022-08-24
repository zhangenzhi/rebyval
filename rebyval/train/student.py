import os
import wandb
from tqdm import trange
from datetime import datetime
import tensorflow as tf
import horovod.tensorflow as hvd

# dataloader
from rebyval.dataloader.factory import dataset_factory

# model
from rebyval.model.factory import model_factory

# others
from rebyval.tools.utils import print_green, print_error, print_normal, check_mkdir, save_yaml_contents
from rebyval.dataloader.utils import glob_tfrecords


class Student(object):
    def __init__(self, student_args, supervisor=None, id=0, dist=False):
        self.args = student_args
        self.supervisor = supervisor
        self.id = id
        self.dist = dist

    def _build_supervisor_from_vars(self, supervisor_info=None):
        model = None
        if supervisor_info != None:
            model = model_factory(supervisor_info)
        return model

    def update_supervisor(self, inputs, labels):
        
        supervisor_opt = tf.keras.optimizers.SGD(0.01)
        supervisor_loss_fn = tf.keras.losses.mae
        flat_vars = []
        for tensor in inputs:
            sum_reduce = tf.math.reduce_sum(tensor, axis=-1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(1, -1)))
        inputs = tf.concat(flat_vars, axis=1)
        
        with tf.GradientTape() as tape:
            predictions = self.supervisor(inputs)
            loss = supervisor_loss_fn(labels, predictions)
            gradients = tape.gradient(
                loss, self.supervisor.trainable_variables)
            supervisor_opt.apply_gradients(
                zip(gradients, self.supervisor.trainable_variables))

    def _build_enviroment(self, devices='0'):
        if self.dist:
            hvd.init()
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        else:
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
        if self.dist:
            optimizer.learning_rate = optimizer.learning_rate * hvd.size()
            optimizer.momentum=0.9
            optimizer.nestrov = False
            optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

    def _build_logger(self):
        logdir = "tensorboard/" + \
            "student-{}-".format(self.id) + \
            datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args['log_path'], logdir)
        if self.dist:
            if hvd.local_rank() == 0:
                check_mkdir(logdir)
        else:
            check_mkdir(logdir)
        logger = tf.summary.create_file_writer(logdir)
        # self.wb = wandb.init(config=self.args, project=self.args['context']['name'], name="student-{}-".format(self.id)+datetime.now().strftime("%Y%m%d-%H%M%S"))
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
        save_path = os.path.join(
            self.valid_args['model_dir'], self.valid_args['save_model']['save_in'])
        save_path = os.path.join(save_path, 'model_{}'.format(name))

        save_msg = '\033[33m[Model Status]: Saving {} model at step:{:08d} in {:}.\033[0m'.format(
            name, self.global_step, save_path)
        print(save_msg)
        self.model.save_weights(save_path, overwrite=True, save_format='tf')

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))

        if self.dist:
            tape = hvd.DistributedGradientTape(tape)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if first_batch:
                hvd.broadcast_variables(self.model.variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels):
        raise NotImplementedError("student need _rebyval_train_step func.")

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        valid_metrics = tf.reduce_mean(self.valid_metrics(labels, predictions))
        self.mv_loss_fn.update_state(loss)
        return loss, valid_metrics

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        test_metrics = tf.reduce_mean(self.test_metrics(labels, predictions))
        self.mtt_loss_fn.update_state(loss)
        return loss, test_metrics
        # return loss

    def train(self, supervisor_info=None):
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        train_args = train_loop_args['train']
        valid_args = train_loop_args['valid']
        test_args = train_loop_args['test']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        
        total_epochs = self.dataloader.info['epochs']  
        if self.dist:
            total_epochs = int(total_epochs/hvd.size())
        train_steps_per_epoch = self.dataloader.info['train_step']

        if supervisor_info != None:
            self.supervisor = self._build_supervisor_from_vars(supervisor_info)

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(total_epochs, desc="Epochs") as e:
            for epoch in e:
                # lr increase
                if train_args["lr_increase"] and epoch<10:
                     self.optimizer.learning_rate = self.optimizer.learning_rate + self.base_lr*(train_args["lr_increase"]-1)/10*4
                     
                # lr decay
                if train_args["lr_decay"]:
                    if epoch == int(0.5*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                    elif epoch == int(0.75*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))

                with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    etr_metrics = []
                    for train_step in t:
                        if self.supervisor == None:
                            if self.dist:
                                first_batch = True if epoch==0 and train_step==0 else False
                                data = train_iter.get_next()
                                train_loss,_, step_train_metric = self._train_step(data['inputs'], data['labels'], first_batch)
                            else:
                                data = train_iter.get_next()
                                train_loss,_, step_train_metric = self._train_step(data['inputs'], data['labels'])
                            etr_metrics.append(step_train_metric)
                        else:
                            data = train_iter.get_next()
                            train_loss = self._rebyval_train_step(data['inputs'], data['labels'], train_step=train_step, epoch=epoch)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        if train_step % valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                for valid_step in v:
                                    v_data = valid_iter.get_next()
                                    valid_loss, valid_metric = self._valid_step(v_data['inputs'], v_data['labels'])
                                    v.set_postfix(sv_loss=valid_loss.numpy())
                                ev_loss = self.mv_loss_fn.result()
                                self.collect_test_metrics(current_state=self.model.trainable_variables,
                                                          metric=ev_loss,
                                                          format=valid_args['weight_space'])
                                # online update supervisor
                                if self.supervisor != None:
                                    self.update_supervisor(self.model.trainable_variables, ev_loss)
                    et_loss = self.mt_loss_fn.result()
                    etr_metric = tf.reduce_mean(etr_metrics)
                
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mtt_loss_fn.reset_states()
                    tt_metrics = []
                    for test_step in t:
                        t_data = test_iter.get_next()
                        t_loss,t_metric = self._test_step(t_data['inputs'], t_data['labels'])
                        t.set_postfix(test_loss=t_loss.numpy())
                        tt_metrics.append(t_metric)
                    ett_loss = self.mtt_loss_fn.result()
                    ett_metric = tf.reduce_mean(tt_metrics)
                    
                e.set_postfix(et_loss=et_loss.numpy(), etr_metric=etr_metric.numpy, ett_loss=ett_loss.numpy(), ett_metric=ett_metric.numpy())
                # train_iter, valid_iter, test_iter = self._reset_dataset()
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_loss", ev_loss, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)
        self.model.summary()

    def run(self, new_student=None, supervisor_info=None, devices='1'):

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

        # self.supervisor = self._build_supervisor_from_vars()
        # with tf.device('GPU:{}'.format(devices)):
        self.train(supervisor_info=supervisor_info)

        self.writter.close()
        print('Finished training student {}'.format(self.id))

        if new_student != None:
            new_student.put(weight_dir)

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
    
    def rl_example(self, experience_buffer):
        configs = {}
        examples = []

        values = list(experience_buffer.values())
        keys = list(experience_buffer.keys())
        sample_value = list(zip(*values))
        samples = [dict(zip(keys, s)) for s in sample_value]
        for s in samples:
            feature = {}
            for feature_name, value in s.items():
                bytes_v = tf.io.serialize_tensor(value).numpy()
                feature[feature_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_v]))
                configs[feature_name] = {'type': 'bytes', 'shape': value.shape.as_list()}
            examples.append(tf.train.Example(features=tf.train.Features(feature=feature)))

        return examples, configs

    def _write_trace_to_tfrecord(self, weights, valid_loss, weight_space=None):

        weight_loss = {'vars': weights, 'valid_loss': valid_loss}

        if weight_space['format'] == 'tensor':
            example, configs = self.tensor_example(weight_loss)
        elif weight_space['format'] == 'sum_reduce':
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
        
    def _write_trail_to_tfrecord(self, experience_buffer):
        
        examples, configs = self.rl_example(experience_buffer)
        for e in examples:
            self.writter.write(e.SerializeToString())
        
        weight_dir = os.path.join(self.args['log_path'], 'weight_space')
        config_path = os.path.join(weight_dir, 'feature_configs.yaml')

        configs['num_of_students'] = len(glob_tfrecords(
            weight_dir, glob_pattern='*.tfrecords'))
        configs['sample_per_student'] = int(self.dataloader.info['train_step'] * self.dataloader.info['epochs'] / self.args['train_loop']['valid']['valid_gap'])
        configs['total_samples'] = configs['sample_per_student'] * \
            configs['num_of_students']
        save_yaml_contents(contents=configs, file_path=config_path)
            