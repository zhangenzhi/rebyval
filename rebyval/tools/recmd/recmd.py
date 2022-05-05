import argparse
import os
import pkg_resources
from colorama import init
from rebyval.tools.utils import print_error
from .launcher import create_experiment, view_experiment
from .constants import DEFAULT_REST_PORT

init(autoreset=True)

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage

    coverage.process_startup()


def rebyval_info(*args):
    if args[0].version:
        try:
            print(pkg_resources.get_distribution('autosparsedl').version)
        except pkg_resources.ResolutionError:
            print_error(
                'Get version failed, please use `pip3 list | grep autosparsedl` to check autosparsedl version!'
            )
    else:
        print(
            'please run "adcmd {positional argument} --help" to see adcmd guidance'
        )


def parse_args():
    '''Definite the arguments users need to follow and input'''
    parser = argparse.ArgumentParser(
        prog='adcmd',
        description='use adcmd command to control autosparsedl experiments')
    parser.add_argument('--version', '-v', action='store_true')
    parser.set_defaults(func=autosparsedl_info)

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # parse start command
    parser_start = subparsers.add_parser('create',
                                         help='create a new experiment')
    parser_start.add_argument('--config',
                              '-c',
                              required=True,
                              dest='config',
                              help='the path of yaml config file')

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    parse_args()
