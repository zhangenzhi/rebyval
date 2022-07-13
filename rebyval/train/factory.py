
from .cifar10_rl_student import Cifar10RLStudent
from .cifar10_student import Cifar10Student
from .cifar10_supervisor import Cifar10Supervisor

from .mnist_student import MnistStudent
from .mnist_supervisor import MnistSupervisor

from .cifar100_student import Cifar100Student
from .cifar100_supervisor import Cifar100Supervisor

from .cifar10_rl_student import Cifar10RLStudent
from .cifar10_rl_supervisor import Cifar10RLSupervisor


class StudentFactory():
    def __init__(self) -> None:
        self.student_list = {'cifar100':Cifar100Student, 'cifar10': Cifar10Student, 'mnist': MnistStudent, 'rl-cifar10':Cifar10RLStudent}

    def __call__(self, student_args, supervisor = None, id = 0):
        return self.get_student(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)

    def get_student(self, student_args, supervisor = None, id = 0):
        if student_args['dataloader'].get('task'):
            student_cls = self.student_list.get('rl-cifar10')
        else:
            student_cls = self.student_list.get(student_args['dataloader']['name'])
        
        return student_cls(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)


class SupervisorFactory():
    def __init__(self) -> None:
        self.supervisor_list = {'cifar100':Cifar100Supervisor, 'cifar10': Cifar10Supervisor, 'mnist': MnistSupervisor, 'rl-cifar10':Cifar10RLSupervisor}
    
    def __call__(self, supervisor_args, student_task='', id = 0):
        return self.get_supervisor(supervisor_args=supervisor_args, student_task=student_task, id=id)

    def get_supervisor(self, supervisor_args = None, student_task='', id = id):
        supervisor_cls = self.supervisor_list.get(student_task['name'])
        if student_task.get('task') == 'RL':
            supervisor_cls = self.supervisor_list.get('rl-'+student_task['name'])
        print(supervisor_cls)
        return supervisor_cls(supervisor_args=supervisor_args, id = id)

student_factory = StudentFactory()
supervisor_factory = SupervisorFactory()