import time
import argparse

from rebyval.tools.utils import *
from rebyval.controller.utils import *
from rebyval.train.trainer import *
from rebyval.controller.base_controller import BaseController


class MultiController(BaseController):
    def __init__(self,yaml_path=None):
        super(MultiController, self).__init__(yaml_path=yaml_path)

    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])

    def main_loop_for_experiment(self):
        target_trainer = self._build_target_trainer()
        target_trainer.run()

    def run(self):
        self._build_enviroment()
        print_green("Start to run!")
        self.main_loop_for_experiment()
        print_green('[Task Status]: Task done! Time cost: {:}')