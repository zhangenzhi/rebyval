import time
import argparse

from rebyval.tools.utils import *
from rebyval.controller.utils import *
from rebyval.train.trainer import *
from rebyval.controller.base_controller import BaseController


class TargetTrainController(BaseController):
    def __init__(self,yaml_path=None):
        super(TargetTrainController, self).__init__(yaml_path=yaml_path)

    def main_loop_for_experiment(self):
        target_trainer = self._build_target_trainer()
        target_trainer.run()

    def run(self):
        self._build_enviroment()
        print_green("Start to run!")
        self.main_loop_for_experiment()
        print_green('[Task Status]: Task done! Time cost: {:}')