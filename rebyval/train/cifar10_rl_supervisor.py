from numpy import reshape
import tensorflow as tf
from tqdm import trange

from rebyval.train.supervisor import Supervisor
from rebyval.tools.utils import print_error
from rebyval.train.utils import ForkedPdb

class Cifar10RLSupervisor(Supervisor):
    def __init__(self, supervisor_args, logger = None, id = 0):
        super(Cifar10RLSupervisor, self).__init__(supervisor_args, logger = logger, id = id)
        
    def __call__(self, inputs):
        s_loss = self.model(inputs, training=False)
        s_loss = tf.squeeze(s_loss)
        return s_loss
            
    def preprocess_weightspace(self, raw_inputs):
        # label
        labels = raw_inputs.pop('Q')
        # labels = tf.reshape(labels,shape=(-1,1))

        # # states
        # inputs = raw_inputs.pop('states')
        # ForkedPdb().set_trace()
        # states & actions
        state = tf.reshape(raw_inputs.pop('states'),shape=(128,-1))
        act_grad = tf.reshape(raw_inputs.pop('act_grads'),shape=(128,-1))
        inputs = tf.concat([state, act_grad],axis=-1)
        
        # inputs = tf.reshape(inputs,shape=(labels.shape[0],-1))
        
        return inputs, labels
        
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            predictions = tf.squeeze(predictions)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
            
        return loss
    
    def update(self, inputs, labels):
        flat_vars = []
        for tensor in inputs:
            sum_reduce = tf.math.reduce_sum(tensor, axis= -1)
            flat_vars.append(tf.reshape(sum_reduce, shape=(-1)))
        inputs = tf.concat(flat_vars, axis=1)
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            predictions = tf.squeeze(predictions)
            loss = self.loss_fn(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        print(loss)
        
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels, valid_step = 0, epoch=0):
        try:
            predictions = self.model(inputs, training=False)
            predictions = tf.squeeze(predictions)
            loss = self.loss_fn(labels, predictions)
        except:
            print_error("valid step error.")
        
        with self.logger.as_default():
            step = valid_step+epoch*self.dataloader.info['valid_step']
            tf.summary.scalar("valid_loss", loss, step=step)
            
        return loss

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels, test_step=0):
        try:
            predictions = self.model(inputs, training=True)
            predictions = tf.squeeze(predictions)
            loss = self.loss_fn(labels, predictions)
        except:
            print_error("test step error.")
            raise 
        
        with self.logger.as_default():
            tf.summary.scalar("test_loss", loss, step=test_step)
        
        return loss

    def train(self):
        
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
        metric_name = self.args['metrics']['name']
        self.metrics[metric_name].reset_states()

        # train, valid, test
        # tqdm update, logger
        with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
            self.mloss_fn.reset_states()
            for epoch in e:
                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    for train_step in t:
                        data = train_iter.get_next()
                        inputs, labels = self.preprocess_weightspace(data)
                        # ForkedPdb().set_trace()
                        train_loss = self._train_step(inputs, labels, train_step=train_step, epoch=epoch)
                        self.mloss_fn.update_state(train_loss)
                        t.set_postfix(st_loss=train_loss.numpy())
                    et_loss = self.mloss_fn.result()
                        
                # valid
                with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                    self.mloss_fn.reset_states()
                    for valid_step in v:
                        data = valid_iter.get_next()
                        inputs,labels = self.preprocess_weightspace(data)
                        valid_loss = self._valid_step(inputs, labels,
                                                    valid_step=valid_step, epoch=epoch)
                        self.mloss_fn.update_state(valid_loss)
                        v.set_postfix(sv_loss=valid_loss.numpy())
                    ev_loss = self.mloss_fn.result()
                    
                # epoch info
                e.set_postfix(et_loss=et_loss.numpy(), ev_loss=ev_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("epoch_train_loss", et_loss, step=self.dataloader.info['epochs']*self.id+epoch)
                    tf.summary.scalar("epoch_valid_loss", ev_loss, step=self.dataloader.info['epochs']*self.id+epoch)
        
        with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
            for test_step in t:
                data = test_iter.get_next()
                inputs,labels = self.preprocess_weightspace(data)
                t_loss = self._test_step(inputs, labels, test_step = test_step)
                t.set_postfix(se_loss=t_loss.numpy())