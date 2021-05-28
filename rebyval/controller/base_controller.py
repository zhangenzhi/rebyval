import time
import argparse

from rebyval.tools.utils import *
from rebyval.controller.utils import *
from rebyval.train.trainer import *


class BaseController:
    def __init__(self, yaml_path=None):

        if yaml_path != None:
            print_normal("parse config from python script.")
            self.yaml_configs = get_yml_content(yaml_path)
        else:
            print_normal("parse config from command line.")
            command_args = self._args_parser()
            self.yaml_configs = get_yml_content(command_args.config)

        print_dict(self.yaml_configs)

    def _args_parser(self):
        parser = argparse.ArgumentParser('autosparsedl_config')
        parser.add_argument(
            '--config',
            type=str,
            default='./examples/experiment_configs/template/rebyval.yaml',
            help='yaml config file path')
        args = parser.parse_args()
        return args

    def _build_enviroment(self):

        self.args = self.yaml_configs['experiment']
        check_args_from_yaml_content(self.args)

    def _build_target_trainer(self, rebyval=False):
        target_trainer_args = self.args["target_trainer"]

        try:
            if rebyval:
                target_trainer = TargetTrainer(trainer_args=target_trainer_args,
                                               surrogate_model=self.surrogate_trainer.model)
            else:
                target_trainer = TargetTrainer(trainer_args=target_trainer_args)
        except:
            print_error("build target trainer failed.")
            raise

        return target_trainer

    def _build_surrogate_trainer(self):
        surrogate_trainer = self.args["surrogate_trainer"]
        try:
            surrogate_trainer = SurrogateTrainer(trainer_args=surrogate_trainer)
        except:
            print_error("build suroragte trainer failed.")
            raise

        return surrogate_trainer

    def before_experiment(self):
        pass

    def warmup_stage(self, warmup_steps):
        target_model_samples = warmup_steps['target_model_samples']
        for i in range(target_model_samples):
            target_trainer = self._build_target_trainer()
            target_trainer.run_with_weights_collect()

        # TODO: add validation for training surrogate model
        # try:
        #     valid_weights_pool(self.surrogate_trainer.args)
        # except:
        #     print_error("No weights in the pool")
        #     raise

        # self.surrogate_trainer.run()

    def main_loop_for_experiment(self):
        main_loop_args = self.args['main_loop_control']

        if main_loop_args.get('warmup_stage'):
            self.warmup_stage(main_loop_args['warmup_stage'])

        for j in range(main_loop_args['main_loop_times']):
            for i in range(main_loop_args['target_samples_per_iter']):
                target_trainer = self._build_target_trainer(rebyval=True)
                target_trainer.run_with_weights_collect()

            # TODO: add validation for training surrogate model
            try:
                valid_weights_pool(self.surrogate_trainer.args)
            except:
                print_error("No weights in the pool")
                raise
            self.surrogate_trainer.run_with_refreshed_dataset()

    def run(self):

        self._build_enviroment()

        print_green("build trainer from surrogate nets")
        self.surrogate_trainer = self._build_surrogate_trainer()

        print_green("Start to run!")
        self.main_loop_for_experiment()

        print_green('[Task Status]: Task done! Time cost: {:}')


if __name__ == '__main__':
    BaseController().run()
