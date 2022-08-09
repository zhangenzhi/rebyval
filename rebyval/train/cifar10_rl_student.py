from tqdm import trange
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
        
        self.index_max = 0
        self.act_idx = []
        self.gloabl_train_step = 0
        self.valid_gap = 100
        self.epsilon = 0.5
        
        ## RL
        self.best_metric = 0.5
        self.baseline = 0.1
        self.experience_buffer = {'states':[], 'rewards':[], 'metrics':[], 'actions':[],
                                  'act_grads':[],'E_Q':[], 'steps':[]}
        
        
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
    
    def fix_action(self, t_grad):

        # fixed action with pseudo sgd
        if self.valid_args['weight_space'] == 'sum_reduce':
            flat_grad = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in t_grad], axis=-1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = t_grad[:2]
            last_layer = t_grad[2:]
            reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in first_layer], axis=-1)
            keep_grads =tf.concat([tf.reshape(g,(1,-1)) for g in last_layer], axis=-1)
            flat_grad = tf.concat([reduced_grads,keep_grads], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            flat_grad = tf.concat([tf.reshape(g,(1,-1)) for g in t_grad], axis=-1)
            
        if self.valid_args['weight_space'] == 'sum_reduce':
            flat_var = tf.concat([tf.reshape(tf.reduce_sum(w, axis=-1),(1,-1)) for w in self.model.trainable_variables], axis=-1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = self.model.trainable_variables[:2]
            last_layer = self.model.trainable_variables[2:]
            reduced_vars = tf.concat([tf.reshape(tf.reduce_sum(w, axis=-1),(1,-1)) for w in first_layer], axis=-1)
            keep_vars =tf.concat([tf.reshape(w,(1,-1)) for w in last_layer], axis=-1)
            flat_var = tf.concat([reduced_vars,keep_vars], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            flat_var = tf.concat([tf.reshape(w,(1,-1)) for w in self.model.trainable_variables], axis=-1)
        
        self.action_sample = tf.reshape(tf.constant([0.1,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], dtype=tf.float32),shape=(-1,1))
        scaled_gards = flat_grad * self.action_sample
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        scaled_vars = var_copy - scaled_gards * self.optimizer.learning_rate
        scaled_vars = tf.reshape(scaled_vars,shape=(10,1,-1))
        # select wights with best Q-value
        steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*self.action_sample.shape[0], dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':tf.squeeze(scaled_vars), 'action':scaled_gards,'step':steps}
        self.values = self.supervisor(states_actions)
        return self.action_sample, self.values
    
    def greedy_policy(self, values):
        if self.id%10==0:
            return 1
        else:
            return max(range(len(values)), key=values.__getitem__) 
    
    def e_greedy_policy(self, values):
        
        if self.id%10==0:
            return max(range(len(values)), key=values.__getitem__) 
        
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
                
        # fixed action with pseudo sgd
        if (self.gloabl_train_step %  self.valid_gap )==0:
            self.pre_act = 1.0 if self.gloabl_train_step<self.valid_gap else self.action_sample[self.greedy_policy(self.values)]
            if self.train_args['action'] == 'fix':
                self.action_sample, self.values = self.fix_action(t_grad=t_grad)
            elif self.train_args['action'] == 'neg':
                self.action_sample, self.values = self.neg_action(t_grad=t_grad)
            elif self.train_args['action'] == 'elem':
                self.action_sample, self.values = self.elem_action(t_grad=t_grad)
            else:
                self.action_sample, self.values = self.soft_action(t_grad=t_grad)

            # greedy policy
            if self.train_args['policy'] == 'e_greedy':
                self.index_max = self.e_greedy_policy(self.values)
            else:
                self.index_max = self.greedy_policy(self.values)
            self.act_idx.append(self.index_max) 

        # next state
        if self.train_args['action'] == 'elem':
            act = [a[self.index_max] for a in self.action_sample]
            gradients = [g*a for g,a in zip(t_grad,act)]
            act = tf.concat([tf.reshape(tf.reduce_sum(a, axis=-1),(1,-1)) for a in act], axis=-1)
        else:
            act = self.action_sample[self.index_max]
            alpha = (self.gloabl_train_step %  self.valid_gap)/self.valid_gap
            smoothed_act = (1-alpha)*self.pre_act + alpha*act
            gradients = [g*smoothed_act for g in t_grad]
            act = tf.squeeze(smoothed_act)
        clip_grads = [tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clip_grads, self.model.trainable_variables))
            
        self.mt_loss_fn.update_state(t_loss)
            
        E_q = tf.squeeze(self.values[self.index_max])
        return t_loss, E_q, act, gradients, self.values

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
                            train_loss, act_grad = self._train_step(data['inputs'], data['labels'])
                            action = tf.ones(shape=act_grad.shape, dtype=tf.float32) if self.train_args['action']=='elem' else 1.0
                            E_Q = 0.0
                        else:
                            train_loss, E_Q, action, act_grad, values = self._rl_train_step(data['inputs'], data['labels'])
                            with self.logger.as_default():
                                tf.summary.scalar("E_Q", E_Q, step=self.gloabl_train_step)
                                tf.summary.scalar("action", action, step=self.gloabl_train_step)
                                # tf.summary.histogram("values", values, step=self.gloabl_train_step)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        # Valid
                        if self.gloabl_train_step % self.valid_gap == 0:
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
                                                       E_Q = E_Q,
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
                    
        self.save_experience(q_mode=self.valid_args["q_mode"])
        
    def save_experience(self, q_mode="static", df=0.9):
        
        if q_mode == "TD":
            if self.supervisor == None:
                # baseline without q-net
                s = len(self.experience_buffer['rewards'])
                t_Q = [self.experience_buffer['rewards'][-1]] 
                for i in reversed(range(s-1)):
                    q_value = self.experience_buffer['rewards'][i] + df*t_Q[0]
                    t_Q.insert(0, q_value)
                Q = []
                for i in range(len(t_Q)):
                    values = self.experience_buffer['values'][i]
                    np_values = values.numpy()
                    np_values[1] = t_Q[i].numpy()
                    Q.append(tf.constant(np_values))
                self.experience_buffer['Q'] = Q
            else:
                # boostrap Q value
                s = len(self.experience_buffer['rewards'])
                Q = []
                for i in range(s):
                    e_q = self.experience_buffer['E_Q'][i+1] if i+1!=s else self.experience_buffer['E_Q'][-1]
                    act_q = self.experience_buffer['rewards'][i] + df*e_q
                    values = self.experience_buffer['values'][i]
                    np_values = values.numpy()
                    if self.id%10 != 0:
                        idx = self.act_idx[i]
                    else:
                        idx = 1
                    np_values[idx] = act_q
                    Q.append(tf.reshape(tf.constant(np_values),shape=(-1,1)))
                        
                self.experience_buffer['Q'] = Q
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(max(Q[i])), step=i)
        else:
            s = len(self.experience_buffer['rewards'])
            Q = [self.experience_buffer['rewards'][-1]] 
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
        
        if self.valid_args['weight_space'] == 'sum_reduce':
            state = tf.concat([tf.reshape(tf.math.reduce_sum(w, axis=-1),(1,-1)) for w in weight], axis=1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = weight[:2]
            last_layer = weight[2:]
            reduced_state = tf.concat([tf.reshape(tf.reduce_sum(w, axis=-1),(1,-1)) for w in first_layer], axis=-1)
            keep_state =tf.concat([tf.reshape(w,(1,-1)) for w in last_layer], axis=-1)
            state = tf.concat([reduced_state,keep_state], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            state = tf.concat([tf.reshape(w,(1,-1)) for w in weight], axis=1)
            
        self.experience_buffer['states'].append(state)
        
        self.experience_buffer['metrics'].append(metric)
        
        # reward function
        self.experience_buffer['rewards'].append(metric)
        self.experience_buffer['E_Q'].append(tf.cast(E_Q, tf.float32))
        
        # expect state values 
        if values !=None :
            self.experience_buffer['values'].append(values)
        self.experience_buffer['actions'].append(tf.constant(action[0]))
        
        gradients = action[1]
        if self.valid_args['weight_space'] == 'sum_reduce':
            reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in gradients], axis=-1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = gradients[:2]
            last_layer = gradients[2:]
            reduced_grad = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in first_layer], axis=-1)
            keep_grads =tf.concat([tf.reshape(w,(1,-1)) for w in last_layer], axis=-1)
            reduced_grads = tf.concat([reduced_grad,keep_grads], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            reduced_grads = tf.concat([tf.reshape(g,(1,-1)) for g in gradients], axis=1)
        self.experience_buffer['act_grads'].append(reduced_grads)
        self.experience_buffer['steps'].append(tf.cast(step, tf.float32))
                
                
        