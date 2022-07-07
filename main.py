import os
import multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse

from rebyval.controller.base_controller import BaseController
from rebyval.controller.dist_controller import DistController


if __name__ == '__main__':

    
    mp.set_start_method("spawn")
    BaseController().run()

