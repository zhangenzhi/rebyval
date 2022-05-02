from sklearn import metrics
from tqdm import trange
import tensorflow as tf

# others
import time
from rebyval.model.dnn import DNN
from rebyval.train.student import Student
from rebyval.train.utils import ForkedPdb
from rebyval.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar10Student(Student):
    
    def __init__(self, student_args, supervisor = None, supervisor_vars=None, id = 0):
        super(Cifar10Student, self).__init__(student_args, supervisor, supervisor_vars,id)
        # self.supervisor_vars = supervisor_vars
        # self.supervisor = self._build_supervisor_from_vars()
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, train_step = 0, epoch=0):
    
        try:
            # with tf.GradientTape() as tape:
            train_step_time = time.time()
            predictions = self.model(inputs, training=True)
            # print(time.time() - train_step_time)
                # loss = self.loss_fn(labels, predictions)
                # print(loss)
            # gradients = tape.gradient(loss, self.model.trainable_variables)
            # self.optimizer.apply_gradients(
            #     zip(gradients, self.model.trainable_variables))
        except:
            print_error("train step error")
            raise
        
        # with self.logger.as_default():
        #     step = train_step+epoch*self.dataloader.info['train_step']
        #     # if (step+1)%(50*self.dataloader.info['train_step']) == 0:
        #     #     self.optimizer.learning_rate = self.optimizer.learning_rate * 0.1
        #     tf.summary.scalar("learning_rate", self.optimizer.learning_rate, step=step)
        loss = tf.constant(0)    
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
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _rebyval_train_step(self, inputs, labels, train_step = 0, epoch=0, decay_factor=0.01):
        
        step = train_step+epoch*self.dataloader.info['train_step']
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            if step % self.dataloader.info['train_step'] == 0:
                self.s_loss = self.weightspace_loss(self.model.trainable_variables)
            t_loss = self.loss_fn(labels, predictions)
            if self.id % 10 == 0:
                loss = t_loss
            else:
                loss = t_loss + self.s_loss
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        
        with self.logger.as_default():
        #     tf.summary.scalar("train_loss", t_loss, step=step)
            tf.summary.scalar("surrogate_loss", self.s_loss, step=step)
            
        self.mt_loss_fn.update_state(t_loss)
        
        return t_loss

    def _valid_step(self, inputs, labels, valid_step = 0, epoch=0, weight_space=None):
        try:
            predictions = self.model(inputs, training=False)
            loss = self.loss_fn(labels, predictions)
            self.metrics.update_state(labels, predictions)
            self.mv_loss_fn.update_state(loss)
        except:
            print_error("valid step error")
            raise
          
        # with self.logger.as_default():
        #     step = valid_step+epoch*self.dataloader.info['valid_step']
        #     tf.summary.scalar("valid_loss", loss, step=step)
        return loss
    
    def _test_step(self, inputs, labels, test_step=0):
        try:
            predictions = self.model(inputs, training=False)
            loss = self.loss_fn(labels, predictions)
            self.mv_loss_fn.update_state(loss)
        except:
            print_error("test step error")
            raise
        
        # with self.logger.as_default():
        #     tf.summary.scalar("test_loss", loss, step=test_step)
            
        return loss

    def train(self, new_student=None, supervisor_vars=None):
        
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
        self.metrics.reset_states()
        
        # import pdb
        # pdb.set_trace()
        if supervisor_vars != None:
            # ForkedPdb().set_trace()
            self.supervisor_vars = supervisor_vars
            self.supervisor = self._build_supervisor_from_vars()

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
            for epoch in e:
                epoch_step_time = time.time()
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
                                self._write_trace_to_tfrecord(weights = self.model.trainable_variables, 
                                                              valid_loss = ev_loss,
                                                              weight_space = valid_args['weight_space'])
                    print(time.time() - epoch_step_time)
                    et_loss = self.mt_loss_fn.result()
                    ev_metric = self.metrics.result()
        
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mv_loss_fn.reset_states()
                    for test_step in t:
                        data = test_iter.get_next()
                        t_loss = self._test_step(data['inputs'], data['labels'], test_step = test_step)
                        t.set_postfix(test_loss=t_loss.numpy())
                    ett_loss = self.mv_loss_fn.result()
                    
                e.set_postfix(et_loss=et_loss.numpy(), ev_loss=ev_loss.numpy(), ett_loss=ett_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_loss", ev_loss, step=epoch)
                    tf.summary.scalar("ev_metric", ev_metric, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                
                
        