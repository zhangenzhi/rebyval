from pyexpat import model
import time
import argparse
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from multiprocessing import Pool, Queue, Process

from rebyval.tools.utils import *
from rebyval.dataloader.utils import *
from rebyval.controller.utils import *
from rebyval.train.factory import student_factory, supervisor_factory
from rebyval.train.cifar10_student import Cifar10Student
from rebyval.train.cifar10_supervisor import Cifar10Supervisor


class BaseController:
    def __init__(self, yaml_path=None):

        if yaml_path != None:
            print_normal("parse config from python script.")
            self.yaml_configs = get_yml_content(yaml_path)
        else:
            print_normal("parse config from command line.")
            command_args = self._args_parser()
            self.yaml_configs = get_yml_content(command_args.config)

        
        self.yaml_configs = check_args_from_input_config(self.yaml_configs)

        self._build_enviroment()
        self.queue = Queue(maxsize=100)
        weight_dir = os.path.join(self.log_path, "weight_space")
        if os.path.exists(weight_dir):
            self._student_ids = len(glob_tfrecords(weight_dir, glob_pattern='*.tfrecords'))
        else:
            self._student_ids = 0
        self._supervisor_ids = 0
        
        self.supervisor = self._build_supervisor()

    def _args_parser(self):
        parser = argparse.ArgumentParser('autosparsedl_config')
        parser.add_argument(
            '--config',
            type=str,
            default='./scripts/configs/cifar10/rebyval.yaml',
            help='yaml config file path')
        args = parser.parse_args()
        return args

    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])

    def _build_student(self, supervisor=None):
        student_args = self.args["student"]
        student_args['log_path'] = self.log_path

        student = student_factory(student_args=student_args, 
                                  supervisor = supervisor,
                                  id = self._student_ids)

        self._student_ids += 1
        return student

    def _build_supervisor(self):
        student_args = self.args["student"]
        supervisor_args = self.args["supervisor"]
        supervisor_args['log_path'] = self.log_path

        supervisor = supervisor_factory(supervisor_args=supervisor_args,
                                        student_task= student_args['dataloader'],
                                        id = self._supervisor_ids)

        self._supervisor_ids += 1
        return supervisor
        
    def warmup(self, warmup):
        init_samples = warmup['student_nums']
        supervisor_trains = warmup['supervisor_trains']
        
        if self.context["multi-p"]:
            processes = []
            for i in range(init_samples):
                student = self._build_student()
                p = Process(target = student.run, args=(self.queue,))
                p.start()
                processes.append(p)
                time.sleep(2)
                if (i+1) % 5 == 0:
                    pres = [p.join() for p in processes]
                    processes = []
            new_students = [self.queue.get() for _ in range(self.queue.qsize())]
        else:
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
            if self.context["multi-p"]:
                # mp students with supervisor
                processes = []
                for i in range(main_loop['student_nums']):
                    student = self._build_student()
                    supervisor_vars = [var.numpy() for var in self.supervisor.model.trainable_variables] # but model vars ok
                    self.args["supervisor"]['model']['initial_value'] = supervisor_vars
                    supervisor_info = self.args["supervisor"]['model']
                    p = Process(target = student.run, args=(self.queue, supervisor_info))
                    p.start()
                    processes.append(p)
                    time.sleep(3)
                pres = [p.join() for p in processes]
                new_students = [self.queue.get() for _ in range(self.queue.qsize())]
            else:
                new_students = []
                for i in range(main_loop['student_nums']):
                    student = self._build_student(supervisor=self.supervisor)
                    new_students.append[student.run()]
                    
            # supervisor
            print_green("new_student:{}, welcome!".format(new_students))
            self.supervisor.run(keep_train=True, new_students=new_students)

    def run(self):
        
        print_green("Start to run!")
        
        self.main_loop()

        print_green('[Task Status]: Task done! Time cost: {:}')

    

