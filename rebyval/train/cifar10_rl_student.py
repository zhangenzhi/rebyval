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
        
        # # sample action with pseudo sgd
        # if self.id % 5 == 0:
        #     action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(100,1))
        # else:
        #     if self.gloabl_train_step <= 1000:
        #         action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(100,1))
        #     else:
        #         action_sample = tf.random.uniform(minval=0, maxval=2, shape=(100,1))
                
        # fixed action with pseudo sgd
        if (self.gloabl_train_step % 30) ==0:
            if self.id % 10 == 0:
                self.action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(3,1))
            else:
                # if self.gloabl_train_step <= 1000:
                #     action_sample = tf.reshape(tf.constant([1.0,1.0,1.0], dtype=tf.float32),shape=(-1,1))
                # else:
                #     # ForkedPdb().set_trace()
                self.action_sample = tf.reshape(tf.constant([0.1,1.0,10.0], dtype=tf.float32),shape=(-1,1))
            scaled_gards = flat_grad * self.action_sample
            var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
            scaled_vars = var_copy - scaled_gards * self.optimizer.learning_rate
            # select wights with best Q-value
            # ForkedPdb().set_trace()
            states_actions = {'state':var_copy, 'action':scaled_gards,'step':tf.constant([self.gloabl_train_step,
                                                                                          self.gloabl_train_step,
                                                                                          self.gloabl_train_step])}
            self.values = self.supervisor(states_actions)
        
        # ForkedPdb().set_trace()
        # # fixed actions and Q-net
        # if self.id % 5 == 0:
        #     action_sample = tf.reshape(tf.constant([1.0,1.0,1.0], dtype=tf.float32),shape=(-1,1))
        # else:
        #     action_sample = tf.reshape(tf.constant([0.1,1.0,10.0], dtype=tf.float32),shape=(-1,1))
        # # select wights with best Q-value
        # var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        # values = self.supervisor({'state':var_copy,'action':action_sample})


        index_max = max(range(len(self.values)), key=self.values.__getitem__)

        # next state
        gradients = [g*self.action_sample[index_max] for g in t_grad]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
            
        self.mt_loss_fn.update_state(t_loss)
        
        # ForkedPdb().set_trace()
        reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in gradients], axis=-1)
        
        return t_loss, tf.squeeze(self.values[index_max]), tf.squeeze(self.action_sample[index_max]), reduced_grads, self.values

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
                # Train
                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss, grads = self._train_step(data['inputs'], data['labels'])
                            action = 1.0
                            act_grad = tf.concat([tf.reshape(tf.reduce_sum(g,axis=-1),(1,-1)) for g in grads], axis=-1)
                        else:
                            train_loss, E_Q, action, act_grad, values = self._rl_train_step(data['inputs'], data['labels'])
                            with self.logger.as_default():
                                tf.summary.scalar("E_Q", E_Q, step=self.gloabl_train_step)
                                tf.summary.scalar("action", action, step=self.gloabl_train_step)
                                tf.summary.histogram("values", values, step=self.gloabl_train_step)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        # Valid
                        if self.gloabl_train_step % valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                vv_metrics = []
                                for valid_step in v:
                                    v_data = valid_iter.get_next()
                                    v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                                    v.set_postfix(sv_loss=v_loss.numpy())
                                    vv_metrics.append(v_metrics)
                                ev_loss = self.mv_loss_fn.result()
                                ev_metric = tf.reduce_mean(v_metrics)
                            self.mem_experience_buffer(weight=self.model.trainable_weights, 
                                                       metric=ev_metric, 
                                                       action=(action, act_grad), 
                                                       step=self.gloabl_train_step)
                        self.gloabl_train_step += 1
                    et_loss = self.mt_loss_fn.result()
                
                # Test
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
                    
                e.set_postfix(et_loss=et_loss.numpy(), ett_metric=ett_metric.numpy(), ett_loss=ett_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_metric", ev_metric, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)
        self.model.summary()
        self.save_experience()
                
                
        