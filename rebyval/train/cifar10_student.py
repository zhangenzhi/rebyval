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

class Cifar10Student(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar10Student, self).__init__(student_args, supervisor,id)

        self.action_space = np.random.uniform(low=1.0, high=1.0, size=100)
        self.index_min = 0

        # self.supervisor_vars = supervisor_vars
        # self.supervisor = self._build_supervisor_from_vars()
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, train_step = 0, epoch=0):
    
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
        
        # with self.logger.as_default():
        #     step = train_step+epoch*self.dataloader.info['train_step']
        #     # if (step+1)%(50*self.dataloader.info['train_step']) == 0:
        #     #     self.optimizer.learning_rate = self.optimizer.learning_rate * 0.1
        #     tf.summary.scalar("learning_rate", self.optimizer.learning_rate.current_lr, step=step)
        self.mt_loss_fn.update_state(loss)
        
        return loss
    
    def weightspace_loss(self, weights):
        # label
        flat_vars = []
        for var in weights:
            sum_reduce = tf.math.reduce_sum(var, axis= -1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(-1)))
        inputs = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
        s_loss = self.supervisor(inputs)
        s_loss = tf.squeeze(s_loss)
        return s_loss

    def _rl_train_step(self, inputs, labels, train_step = 0, epoch=0, action_space=[0.01, 1.5]):
        
        step = train_step+epoch*self.dataloader.info['train_step']

        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        norm_gard = [g/(1e-8+tf.norm(g)) for g in t_grad]
        
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in norm_gard]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))

        # sample action
        action_sample = tf.random.uniform(minval=action_space[0], maxval=action_space[1], shape=(100,1))
        scaled_gards = flat_grad * action_sample
        var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
        scaled_vars = var_copy - scaled_gards * 0.1
        values = self.supervisor(scaled_vars)
        index_min = min(range(len(values)), key=values.__getitem__)

        # next state
        gradients = [g*action_sample[index_min] for g in norm_gard]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        with self.logger.as_default():
            tf.summary.scalar("surrogate_loss", values[index_min], step=step)
            tf.summary.scalar("action", action_space[index_min], step=step)
            tf.summary.histogram("values", values, step=step)
            
        self.mt_loss_fn.update_state(t_loss)
        return t_loss

    def _reval_train_step(self, inputs, labels, train_step = 0, epoch=0, actions=[0.0, 2.0]):
        
        step = train_step+epoch*self.dataloader.info['train_step']

        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        norm_gard = [g/(1e-8+tf.norm(g)) for g in t_grad]
        
        # sample actions per 10 steps, add baseline per 10 samples 
        if self.id % 10 != 0:
            if (step+1) % 10 == 0:
                values = []
                self.action_space = np.random.uniform(low=actions[0], high=actions[1], size=100)
                for action in self.action_space:
                    scaled_grad = [g*action for g in norm_gard]
                    n_s = [s - 0.1*g*10 for s,g in zip(self.model.trainable_variables, scaled_grad)]
                    v = self.weightspace_loss(n_s)
                    values.append(v)
                self.index_min = min(range(len(values)), key=values.__getitem__)
                with self.logger.as_default():
                    tf.summary.scalar("surrogate_loss", values[self.index_min], step=step)
                    tf.summary.histogram("values", tf.concat(values, axis=0), step=step)
    
        # next_state
        gradients = [g*self.action_space[self.index_min] for g in norm_gard]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        with self.logger.as_default():
            tf.summary.scalar("action", self.action_space[self.index_min], step=step)

        self.mt_loss_fn.update_state(t_loss)
        return t_loss
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels, train_step = 0, epoch=0, decay_factor=0.1):
        
        step = train_step+epoch*self.dataloader.info['train_step']

        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        if train_step % 100 == 0:
            with tf.GradientTape() as tape_s:
                self.s_loss = self.weightspace_loss(self.model.trainable_variables)
            self.s_grad = tape_s.gradient(self.s_loss, self.model.trainable_variables)

        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
        # gradients = [(s/(1e-12 + tf.norm(s)) + t/(1e-8 + tf.norm(t)))/2 for s,t in zip(self.s_grad,t_grad)]
        gradients = [(s/(1e-12 + tf.norm(s)))*decay_factor+t for s,t in zip(self.s_grad,t_grad)]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        with self.logger.as_default():
            tf.summary.scalar("surrogate_loss", self.s_loss, step=step)
            # if step % self.dataloader.info['train_step'] == 0:
            #     tf.summary.histogram("t_gard_0", t_grad[0], step=step)
            #     tf.summary.histogram("s_gard_0", self.s_grad[0], step=step)
            
        self.mt_loss_fn.update_state(t_loss)
        
        return t_loss

    def _valid_step(self, inputs, labels, valid_step = 0, epoch=0, weight_space=None):
        
        # step = valid_step + epoch*self.dataloader.info['valid_step']

        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        self.mv_loss_fn.update_state(loss)
        # with self.logger.as_default():
        #     tf.summary.scalar("v_loss", loss, step=step)
        return loss
    
    def _test_step(self, inputs, labels, test_step=0):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        self.test_metrics.update_state(labels, predictions)
        self.mtt_loss_fn.update_state(loss)
        return loss

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
        
        # metrics reset
        # self.test_metrics.reset_states()
        
        if supervisor_info != None:
            # self.supervisor_info = supervisor_info
            self.supervisor = self._build_supervisor_from_vars(supervisor_info)

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
            for epoch in e:

                # lr decay
                if epoch/self.dataloader.info['epochs'] == 0.5:
                    self.optimizer.learning_rate = self.optimizer.learning_rate*0.5
                    print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                elif epoch/self.dataloader.info['epochs'] == 0.75:
                    self.optimizer.learning_rate = self.optimizer.learning_rate*0.5
                    print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))

                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss = self._train_step(data['inputs'], data['labels'], 
                                                        train_step=train_step, epoch=epoch)
                        else:
                            train_loss = self._rebyval_train_step(data['inputs'], data['labels'], 
                                                        train_step=train_step, epoch=epoch)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        if train_step % valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                for valid_step in v:
                                    data = valid_iter.get_next()
                                    valid_loss = self._valid_step(data['inputs'], data['labels'],
                                                                valid_step=valid_step, epoch=epoch, 
                                                                weight_space=valid_args['weight_space'])
                                    v.set_postfix(sv_loss=valid_loss.numpy())
                                ev_loss = self.mv_loss_fn.result()
                                # online update supervisor
                                if self.supervisor != None:
                                    self.update_supervisor(self.model.trainable_variables, ev_loss)
                                # self._write_trace_to_tfrecord(weights = self.model.trainable_variables, 
                                #                               valid_loss = ev_loss,
                                #                               weight_space = valid_args['weight_space'])
                    et_loss = self.mt_loss_fn.result()
                
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mtt_loss_fn.reset_states()
                    self.test_metrics.reset_states()
                    for test_step in t:
                        data = test_iter.get_next()
                        t_loss = self._test_step(data['inputs'], data['labels'], test_step = test_step)
                        t.set_postfix(test_loss=t_loss.numpy())
                    ett_loss = self.mtt_loss_fn.result()
                    ett_metric = self.test_metrics.result()
                    
                e.set_postfix(et_loss=et_loss.numpy(), ev_loss=ev_loss.numpy(), ett_loss=ett_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_loss", ev_loss, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)
                
                
        