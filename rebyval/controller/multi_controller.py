import time
import tensorflow as tf
import multiprocessing as mp
from torch.multiprocessing import Pool, Queue, Process

from rebyval.tools.utils import *
from rebyval.train.utils import ForkedPdb
from rebyval.controller.utils import *
from rebyval.controller.base_controller import BaseController


class MultiController(BaseController):
    def __init__(self, yaml_configs):
        super(MultiController, self).__init__(yaml_configs=yaml_configs)
        self.queue = Queue(maxsize=100)

    def _build_enviroment(self):
        mp.set_start_method("spawn")

        self.gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in self.gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])

    def gpu_dispatch(self, student):
        num_gpus = len(self.gpus)
        if student.id % num_gpus == 0:
            avail_gpu = student.id % num_gpus
        return avail_gpu
        
    def warmup(self, warmup):
        init_samples = warmup['student_nums']
        supervisor_trains = warmup['supervisor_trains']
        
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
   
        for j in range(supervisor_trains):
            keep_train = False if j == 0 else True
            self.supervisor.run(keep_train=keep_train, new_students=[])
            
    def main_loop(self):

        main_loop = self.args['main_loop']

        # init weights pool
        if 'warmup' in main_loop:
            self.warmup(main_loop['warmup'])

        # main loop
        total_students = [self._build_student() for i in range(main_loop['student_nums']*main_loop['nums'])]

        for j in range(main_loop['nums']):
            # mp students with supervisor
            processes = []
            ForkedPdb().set_trace()
            for i in range(main_loop['student_nums']):
                student = total_students.pop(0)
                supervisor_vars = [var.numpy() for var in self.supervisor.model.trainable_variables] # but model vars ok
                self.args["supervisor"]['model']['initial_value'] = supervisor_vars
                supervisor_info = self.args["supervisor"]['model']
                
                p = StudentProcess(student=student, new_student=self.queue, supervisor_info=supervisor_info)
                # p = Process(target = student.run, args=(self.queue, supervisor_info))
                p.start()
                processes.append(p)
                time.sleep(3)
            pres = [p.join() for p in processes]
            new_students = [self.queue.get() for _ in range(self.queue.qsize())]
                    
            # supervisor
            print_green("new_student:{}, welcome!".format(new_students))
            self.supervisor.run(keep_train=True, new_students=new_students)

    def run(self):
        print_green("Start to run!")        
        self.main_loop()
        print_green('[Task Status]: Task done!')

class StudentProcess(Process):
    def __init__(self, student, new_student=None, supervisor_info=None):
        super().__init__()
        print("Init Student Process.")
        self.student = student 
        self.new_student = new_student
        self.supervisor_info = supervisor_info
        return

    def run(self):
        print("Hello World!")
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.gpus = tf.config.experimental.list_physical_devices("GPU")
        print(self.gpus)
        self.student.run(new_student=self.new_student, supervisor_info=self.supervisor_info)
