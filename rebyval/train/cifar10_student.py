
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
    # def _valid_step(self, inputs, labels):
    #     predictions = self.model(inputs, training=False)
    #     loss = self.loss_fn(labels, predictions)
    #     self.mv_loss_fn.update_state(loss)
    #     return loss

    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    # def _test_step(self, inputs, labels):
    #     predictions = self.model(inputs, training=False)
    #     loss = self.loss_fn(labels, predictions)
    #     test_metrics = tf.reduce_mean(self.test_metrics(labels, predictions))
    #     self.mtt_loss_fn.update_state(loss)
    #     return loss, test_metrics

    # def train(self, new_student=None, supervisor_info=None):
        
    #     # parse train loop control args
    #     train_loop_args = self.args['train_loop']
    #     train_args = train_loop_args['train']
    #     valid_args = train_loop_args['valid']
    #     test_args = train_loop_args['test']

    #     # dataset train, valid, test
    #     train_iter = iter(self.train_dataset)
    #     valid_iter = iter(self.valid_dataset)
    #     test_iter = iter(self.test_dataset)
        
        
    #     if supervisor_info != None:
    #         # self.supervisor_info = supervisor_info
    #         self.supervisor = self._build_supervisor_from_vars(supervisor_info)

    #     # train, valid, write to tfrecords, test
    #     # tqdm update, logger
    #     with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
    #         for epoch in e:
    #             # lr decay
    #             if int(self.dataloader.info['epochs']*0.5) <= epoch < int(self.dataloader.info['epochs']*0.75):
    #                 self.optimizer.learning_rate = 0.0001
    #                 print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
    #             elif int(self.dataloader.info['epochs']*0.75) <= epoch < int(self.dataloader.info['epochs']*0.95):
    #                 self.optimizer.learning_rate = 0.00001
    #                 print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
    #             elif int(self.dataloader.info['epochs']*0.95) <= epoch:
    #                 self.optimizer.learning_rate = 0.000001
    #                 print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))

    #             with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
    #                 self.mt_loss_fn.reset_states()
    #                 for train_step in t:
    #                     data = train_iter.get_next()
    #                     if self.supervisor == None:
    #                         train_loss = self._train_step(data['inputs'], data['labels'])
    #                     else:
    #                         train_loss = self._rebyval_train_step(data['inputs'], data['labels'], 
    #                                                     train_step=train_step, epoch=epoch)
    #                     t.set_postfix(st_loss=train_loss.numpy())
                        
    #                     if train_step % valid_args['valid_gap'] == 0:
    #                         with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
    #                             self.mv_loss_fn.reset_states()
    #                             for valid_step in v:
    #                                 v_data = valid_iter.get_next()
    #                                 valid_loss = self._valid_step(v_data['inputs'], v_data['labels'],
    #                                                             valid_step=valid_step, epoch=epoch, 
    #                                                             weight_space=valid_args['weight_space'])
    #                                 v.set_postfix(sv_loss=valid_loss.numpy())
    #                             ev_loss = self.mv_loss_fn.result()
    #                             # online update supervisor
    #                             if self.supervisor != None:
    #                                 self.update_supervisor(self.model.trainable_variables, ev_loss)
    #                             # self._write_trace_to_tfrecord(weights = self.model.trainable_variables, 
    #                             #                               valid_loss = ev_loss,
    #                             #                               weight_space = valid_args['weight_space'])
    #                 et_loss = self.mt_loss_fn.result()
                
    #             with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
    #                 self.mtt_loss_fn.reset_states()
    #                 tt_metrics = []
    #                 for test_step in t:
    #                     # self.test_metrics.reset_states()
    #                     t_data = test_iter.get_next()
    #                     t_loss,t_metric = self._test_step(t_data['inputs'], t_data['labels'])
    #                     t.set_postfix(test_loss=t_loss.numpy())
    #                     tt_metrics.append(t_metric)
    #                 ett_loss = self.mtt_loss_fn.result()
    #                 ett_metric = tf.reduce_mean(tt_metrics)
                    
    #             e.set_postfix(et_loss=et_loss.numpy(), ett_metric=ett_metric.numpy(), ett_loss=ett_loss.numpy())
    #             with self.logger.as_default():
    #                 tf.summary.scalar("et_loss", et_loss, step=epoch)
    #                 tf.summary.scalar("ev_loss", ev_loss, step=epoch)
    #                 tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
    #                 tf.summary.scalar("ett_metric", ett_metric, step=epoch)
    #     self.model.summary()
                
                
        