import os
from time import sleep
import multiprocessing as mp
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse

from rebyval.controller.base_controller import BaseController


if __name__ == '__main__':
    mp.set_start_method('spawn')
    BaseController().run()

