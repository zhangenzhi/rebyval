# Copyright 2020 Xiaomi

import time
import argparse

from autosparsedl.tools.adcmd.common_utils \
    import get_format_time, get_json_content, get_yml_content
from rebyval.tools.reserved_args import DEFAULT_TUNER_ARGS
from rebyval.train.trainer import *


# Fixed build-in objective function for shell start-up
def objective(args):
    if args.dataset == 'cvr':
        trainer = CVRTrainer(args=args)
    elif args.dataset == 'ctr':
        trainer = CTRTrainer(args=args)
    else:
        raise NotImplementedError('such dataset and trainer does not exist.')
    res_dict = trainer.run()
    return res_dict


class BaseController:
    def __init__(self, args):
        self.dict_args = args
        self.controller_args = args["controller"]

        self.set_default_args(self.args, DEFAULT_TUNER_ARGS)

        self.trainer = None
        self.tuner = self._build_tuner()

    def check_and_set_args(self, args, default_args):
        for a in default_args:
            if a not in args.keys():
                # print('Using default args >> {:}: {:}'.format(
                #    a, default_args[a]))
                args[a] = default_args[a]

    def _build_trainer(self):
        trainer_args = ToArgs(**self.args.trainer)
        from autosparsedl.train.trainer import CVRTrainer, CTRTrainer
        if trainer_args.dataset == 'cvr':
            trainer = CVRTrainer(args=trainer_args)
        elif trainer_args.dataset == 'ctr':
            trainer = CTRTrainer(args=trainer_args)
        else:
            raise NotImplementedError('such dataset and trainer does not exist.')
        return trainer

    def _valid_(self):
        pass

    def run(self):
        print("Start to run!")
        s_time = time.time()
        tuner_args = ToArgs(**{**self.args.tuner})

        if tuner_args.algo == 'ea':
            self.tuner.run()
        elif tuner_args.algo == "fulltrain":
            self.trainer.run()
        else:
            # for hperopt tuner
            hpt_search_space = self.search_space.getHptSearchSpace()
            print("Start tuner search")
            self.tuner.search(hpt_search_space)
            if tuner_args.fulltrain:
                self.tuner.fulltrain()

        c_time = time.time() - s_time
        f_time = get_format_time(c_time)
        print('\033[32m[Task Status]: Task done! Time cost: {:}\033[0m' \
              .format(f_time))
        return 'Success!'
