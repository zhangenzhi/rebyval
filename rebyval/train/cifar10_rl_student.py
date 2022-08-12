from tqdm import trange

import wandb
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

# others
import time
from rebyval.train.student import Student
from rebyval.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar10RLStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar10RLStudent, self).__init__(student_args, supervisor,id)
        
        self.act_idx = []
        self.gloabl_train_step = 0
        self.valid_gap = 100
        
        ## RL
        self.best_metric = 0.5
        self.epsilon = 0.5 + self.id*0.001/2.0
        self.action_sample = [0.1,0.5,1.0,2.5,5.0]
        self.index_max = int(len(self.action_sample))
        self.baseline = 0.1
        self.experience_buffer = {'states':[], 'rewards':[], 'metrics':[], 'actions':[], 'values':[],
                                  'act_grads':[],'E_Q':[], 'steps':[]}
    
    def reduced_space(self, state):
        if self.valid_args['weight_space'] == 'sum_reduce':
            flat_state = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in state], axis=-1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = state[:2]
            last_layer = state[2:]
            reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in first_layer], axis=-1)
            keep_grads =tf.concat([tf.reshape(g,(1,-1)) for g in last_layer], axis=-1)
            flat_state = tf.concat([reduced_grads,keep_grads], axis=-1)
        elif self.valid_args['weight_space'] == 'norm_reduce':
            flat_state = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1))/tf.norm(g) for g in state], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            flat_state = tf.concat([tf.reshape(g,(1,-1)) for g in state], axis=-1)
        return flat_state
        
    def elem_action(self, t_grad, num_act=1000):
        # fixed action with pseudo sgd
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))

        if self.id % 10 == 0:
            self.action_sample = []
            for g in t_grad:
                shape = g.shape
                self.action_sample.append( tf.random.uniform(minval=1.0, maxval=1.0, shape=[num_act]+list(shape)))
        else:
            self.action_sample = []
            for g in t_grad:
                shape = g.shape
                self.action_sample.append( tf.random.uniform(minval=0.1, maxval=5.0, shape=[num_act]+list(shape)))

        scaled_grads = [g*a for g, a in zip(t_grad, self.action_sample)]
        flat_scaled_gards = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(num_act, -1)) for g in scaled_grads]
        flat_scaled_gards = tf.concat(flat_scaled_gards, axis=1)
        
        var_copy = tf.tile(flat_var, [flat_scaled_gards.shape.as_list()[0], 1])

        # select wights with best Q-value
        steps = tf.reshape(tf.constant([self.gloabl_train_step/1000]*num_act, dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':var_copy, 'action':flat_scaled_gards,'step':steps}
        self.values = self.supervisor(states_actions)
        return self.action_sample, self.values

    def soft_action(self, t_grad, num_act=64):
        # fixed action with pseudo sgd
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        if self.id % 10 == 0:
            self.action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(num_act,1))
        else:
            self.action_sample = tf.random.uniform(minval=0.01, maxval=5.0, shape=(num_act,1))
        scaled_gards = flat_grad * self.action_sample
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        # select wights with best Q-value
        steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*self.action_sample.shape[0], dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':var_copy, 'action':scaled_gards,'step':steps}
        self.values = self.supervisor(states_actions)
        return self.action_sample, self.values
    
    def neg_action(self, t_grad):
        # fixed action with pseudo sgd
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        if self.id % 10 == 0:
            self.action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(10,1))
        else:
            self.action_sample = tf.reshape(tf.constant([0.01,0.1,1.0,1.5,2.0,-0.01,-0.1,-1.0,-1.5,-2.0], dtype=tf.float32),shape=(-1,1))
        scaled_gards = flat_grad * self.action_sample
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        # select wights with best Q-value
        steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*self.action_sample.shape[0], dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':var_copy, 'action':scaled_gards,'step':steps}
        self.values = self.supervisor(states_actions)
        return self.action_sample, self.values
    
    def fix_n_action(self):
        flat_var = self.reduced_space(self.model.trainable_variables)
        state = tf.reshape(tf.concat(flat_var, axis=0), (1,1,-1))
        self.values = self.supervisor({'state':state})
        return self.action_sample, self.values
    
    def fix_action(self, t_grad):
        action_samples = tf.reshape(tf.constant(self.action_sample, dtype=tf.float32),shape=(-1,1))
        # fixed action with pseudo sgd
        flat_grad = self.reduced_space(t_grad)
        flat_var = self.reduced_space(self.model.trainable_variables)
        
        scaled_gards = flat_grad * action_samples
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        scaled_vars = var_copy - scaled_gards * self.optimizer.learning_rate
        scaled_vars = tf.reshape(scaled_vars,shape=(action_samples.shape.as_list()[0],1,-1))
        # select wights with best Q-value
        steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*action_samples.shape[0], dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':tf.squeeze(scaled_vars), 'action':scaled_gards,'step':steps}
        self.values = self.supervisor(states_actions)
        return action_samples, self.values
    
    def e_greedy_policy(self, values):
        roll = np.random.uniform()
        if roll < self.epsilon:
            return np.random.randint(len(values))
        else:
            return max(range(len(values)), key=values.__getitem__) 
    
    def _rl_train_step(self, inputs, labels):

        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        self.mt_loss_fn.update_state(t_loss)
                
        # fixed action with pseudo sgd
        if (self.gloabl_train_step %  self.valid_gap )==0:
            if self.train_args['action'] == 'fix':
                _, values = self.fix_action(t_grad=t_grad)
            elif self.train_args['action'] == 'fix_n':
                _, values = self.fix_n_action()

            # greedy policy
            self.index_max = self.e_greedy_policy(values)
            E_Q = tf.squeeze(values[self.index_max])
            self.act_idx.append(self.index_max) 

        # next state
        act = self.action_sample[self.index_max]
        gradients = [g*act for g in t_grad]
        clip_grads = [tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clip_grads, self.model.trainable_variables))
            
        return t_loss, E_Q, act, t_grad, values
    
    def train(self, new_student=None, supervisor_info=None):
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        self.train_args = train_loop_args['train']
        self.valid_args = train_loop_args['valid']
        self.valid_gap = self.valid_args['valid_gap']
        self.test_args = train_loop_args['test']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        if supervisor_info != None:
            self.supervisor = self._build_supervisor_from_vars(supervisor_info)
        
        total_epochs = self.dataloader.info['epochs']
        if self.dist:
            total_epochs = int(total_epochs/hvd.size())

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(total_epochs, desc="Epochs") as e:
            for epoch in e:
                
                # lr decay
                if self.train_args["lr_decay"]:
                    if epoch == int(0.5*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                    elif epoch == int(0.75*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                        
                # Train
                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss, t_grad = self._train_step(data['inputs'], data['labels'])
                            action = 1.0
                            idx = int(len(self.action_sample)/2)
                            values = tf.ones(shape=(len(self.action_sample)), dtype=tf.float32)
                            E_Q = -1.0
                        else:
                            train_loss, E_Q, action, t_grad, values = self._rl_train_step(data['inputs'], data['labels'])
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        # Valid && Evaluate
                        if self.gloabl_train_step % self.valid_gap == 0:
                            ev_metric = self.evaluate(valid_iter=valid_iter, E_Q = E_Q, values = values, action=action, t_grad = t_grad)
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
                # wandb log
                self.wb.log({"ett_metric":ett_metric, "et_loss":et_loss, "ev_metric":ev_metric, "ett_mloss":ett_loss})     
                
        self.save_experience(q_mode=self.valid_args["q_mode"])
        self.wb.finish()
        
    def save_experience(self, q_mode="static", df=0.9):
        
        # import pdb
        # pdb.set_trace()
        
        if q_mode == "TD-NQ":
            if self.supervisor == None:
                # baseline without q-net
                values = self.experience_buffer['values']
                self.experience_buffer['Q'] = values
            else:
                # boostrap Q value
                values = self.experience_buffer['values']
                rewards = len(self.experience_buffer['rewards'])
                for i in range(len(rewards)):
                    np_values = values[i].numpy()
                    e_q = rewards[i] + df * values[i][self.act_idx[i]] 
                    np_values[self.act_idx[i]] = e_q
                    values[i] = tf.reshape(tf.constant(np_values), shape=values[i].shape)
                self.experience_buffer['Q'] = values
                
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(max(Q[i])), step=i)
                    
        elif q_mode == "TD":
            s = len(self.experience_buffer['rewards'])
            Q = []
            for i in range(s):
                q_value = self.experience_buffer['rewards'][i] + df*self.experience_buffer['E_Q'][i]
                Q.append(q_value)
            self.experience_buffer['Q'] = [v for v in Q]
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)
                    
        elif q_mode == 'static':
            s = len(self.experience_buffer['rewards'])
            # Q = [self.experience_buffer['rewards'][-1]] 
            Q = [tf.constant(10.0,shape=self.experience_buffer['rewards'][-1].shape)] 
            for i in reversed(range(s-1)):
                q_value = self.experience_buffer['rewards'][i] + df*Q[0]
                Q.insert(0, q_value)
            self.experience_buffer['Q'] = [v for v in Q]
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)
        
        self._write_trail_to_tfrecord(self.experience_buffer)

        return self.best_metric - self.baseline/10 
    
    def mem_experience_buffer(self, weight, metric, action, values=None, E_Q=1.0, step=0):
            
        # state
        reduced_state = self.reduced_space(weight)
        self.experience_buffer['states'].append(reduced_state)
        
        # reward function
        self.experience_buffer['metrics'].append(metric)
        self.experience_buffer['rewards'].append(metric)
        self.experience_buffer['E_Q'].append(tf.cast(E_Q, tf.float32))
        
        # expect state values 
        if values !=None :
            self.experience_buffer['values'].append(values)
        
        # action
        t_grad = action[1]
        reduced_grads = self.reduced_space(t_grad)
        self.experience_buffer['act_grads'].append(reduced_grads)
        self.experience_buffer['actions'].append(tf.constant(action[0]))
        self.experience_buffer['steps'].append(tf.cast(step, tf.float32))
        
        
    def evaluate(self, valid_iter, E_Q, values, action, t_grad):
        
        # warmup sample 
        if self.supervisor == None and self.valid_args["q_mode"] == "TD_NQ":
            raw_values = []
            back_grad = [-action*g for g in t_grad]
            self.optimizer.apply_gradients(zip(back_grad, self.model.trainable_variables))
            
            for i in range(len(self.action_sample)):
                grad = [t_g * self.action_sample[i] for t_g in t_grad]
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                self.mv_loss_fn.reset_states()
                vv_metrics = []
                for valid_step in range(self.dataloader.info['valid_step']):
                    v_data = valid_iter.get_next()
                    v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                    vv_metrics.append(v_metrics)
                ev_loss = self.mv_loss_fn.result()
                ev_metric = tf.reduce_mean(v_metrics)
                raw_values.append(ev_metric)
                re_grad = [-g for g in grad]
                self.optimizer.apply_gradients(zip(re_grad, self.model.trainable_variables))
            values = [10.0 * v for v in raw_values]
            
            fore_grad =  [action*g for g in t_grad]
            self.optimizer.apply_gradients(zip(fore_grad, self.model.trainable_variables))
            ev_metric = values[int(len(self.action_sample)/2)]
            
        else:
            self.mv_loss_fn.reset_states()
            vv_metrics = []
            for valid_step in range(self.dataloader.info['valid_step']):
                v_data = valid_iter.get_next()
                v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                vv_metrics.append(v_metrics)
            ev_loss = self.mv_loss_fn.result()
            ev_metric = tf.reduce_mean(v_metrics)
        
        # save sample
        if self.valid_args["q_mode"] == "TD":
            E_Q = 10.0 * ev_metric if E_Q < 0.0 else E_Q
        elif self.valid_args["q_mode"] == "TD_NQ":
            E_Q = 10.0 * ev_metric if E_Q < 0.0 else E_Q
        elif self.valid_args["q_mode"] == "static":
            E_Q = E_Q
        self.wb.log({"E_Q":E_Q, "action":action, "values":values})  # wandb log
        self.mem_experience_buffer(weight=self.model.trainable_weights, 
                                metric=ev_metric, 
                                action=(action, t_grad), 
                                E_Q = E_Q,
                                values= values,
                                step=self.gloabl_train_step)
        return ev_metric
                
                
        