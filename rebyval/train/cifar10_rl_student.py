from matplotlib.pyplot import sca
from sklearn import metrics
from tqdm import trange
import numpy as np
import tensorflow as tf

# others
import time
from rebyval.train.student import Student
from rebyval.train.utils import ForkedPdb
from rebyval.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar10RLStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar10RLStudent, self).__init__(student_args, supervisor,id)

        self.action_space = np.random.uniform(low=1.0, high=1.0, size=100)
        self.index_min = 0
        self.gloabl_train_step = 0
    
    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
    
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            norm_gard = gradients
            # norm_gard = [g/(1e-8+tf.norm(g)) for g in gradients]
            self.optimizer.apply_gradients(
                zip(norm_gard, self.model.trainable_variables))
        except:
            print_error("train step error")
            raise
        self.mt_loss_fn.update_state(loss)
        
        return loss
    
    def _rl_train_step(self, inputs, labels):

        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))

        # sample action with pseudo sgd
        action_sample = tf.random.uniform(minval=0, maxval=2, shape=(100,1))
        scaled_gards = flat_grad * action_sample
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        scaled_vars = var_copy - scaled_gards * self.optimizer.learning_rate
        values = self.supervisor(scaled_vars)
        index_min = min(range(len(values)), key=values.__getitem__)

        # next state
        gradients = [g*action_sample[index_min] for g in t_grad]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
            
        self.mt_loss_fn.update_state(t_loss)
        return t_loss, values[index_min], action_sample[index_min], values
    
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

    def train(self, new_student=None, supervisor_info=None):
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        train_args = train_loop_args['train']
        valid_args = train_loop_args['valid']
        test_args = train_loop_args['test']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        if supervisor_info != None:
            self.supervisor = self._build_supervisor_from_vars(supervisor_info)

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
            for epoch in e:

                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        self.gloabl_train_step += 1
                        
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss = self._train_step(data['inputs'], data['labels'])
                        else:
                            train_loss, surrogate_loss, action, values = self._rl_train_step(data['inputs'], data['labels'])
                            with self.logger.as_default():
                                tf.summary.scalar("surrogate_loss", surrogate_loss, step=self.gloabl_train_step)
                                tf.summary.scalar("action", action, step=self.gloabl_train_step)
                                tf.summary.histogram("values", values, step=self.gloabl_train_step)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        if train_step % valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                vv_metrics = []
                                for valid_step in v:
                                    v_data = valid_iter.get_next()
                                    v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'],
                                                                valid_step=valid_step, epoch=epoch, 
                                                                weight_space=valid_args['weight_space'])
                                    v.set_postfix(sv_loss=v_loss.numpy())
                                    vv_metrics.append(v_metrics)
                                ev_loss = self.mv_loss_fn.result()
                    et_loss = self.mt_loss_fn.result()
                
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mtt_loss_fn.reset_states()
                    tt_metrics = []
                    for test_step in t:
                        # self.test_metrics.reset_states()
                        t_data = test_iter.get_next()
                        t_loss,t_metric = self._test_step(t_data['inputs'], t_data['labels'])
                        t.set_postfix(test_loss=t_loss.numpy())
                        tt_metrics.append(t_metric)
                    ett_loss = self.mtt_loss_fn.result()
                    ett_metric = tf.reduce_mean(tt_metrics)
                    
                e.set_postfix(et_loss=et_loss.numpy(), ev_loss=ev_loss.numpy(), ett_loss=ett_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_loss", ev_loss, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)
        self.model.summary()
                
                
        