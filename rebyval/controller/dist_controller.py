import horovod.tensorflow as hvd

import tensorflow as tf

from rebyval.tools.utils import *
from rebyval.controller.utils import *
from rebyval.controller.base_controller import BaseController


class DistController(BaseController):
    def __init__(self, yaml_configs):
        super(DistController, self).__init__(yaml_configs=yaml_configs)

    def _build_enviroment(self):
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
            
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])

    def main_loop_for_experiment(self):
        target_trainer = self._build_target_trainer()
        target_trainer.run()

    def warmup(self, warmup):
        init_samples = warmup['student_nums']
        supervisor_trains = warmup['supervisor_trains']
        
        for i in range(init_samples):
            student = self._build_student()
            student.run()
        
        for j in range(supervisor_trains):
            keep_train = False if j == 0 else True
            self.supervisor.run(keep_train=keep_train, new_students=[])

    def main_loop(self):

        main_loop = self.args['main_loop']

        # init weights pool
        if 'warmup' in main_loop:
            self.warmup(main_loop['warmup'])

        # main loop
        for j in range(main_loop['nums']):
            new_students = []
            for i in range(main_loop['student_nums']):
                student = self._build_student(supervisor=self.supervisor)
                new_students.append(student.run())
                    
            # supervisor
            print_green("new_student:{}, welcome!".format(new_students))
            self.supervisor.run(keep_train=True, new_students=new_students)