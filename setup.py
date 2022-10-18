import sys
import os
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit("Sorry, Python < 3.5 is not supported.")

install_dependencies = [
    'networkx==2.5', 'tqdm==4.51.0', 'PyYAML==5.4.1', 'cloudpickle==1.6.0',
    'future==0.18.2', 'ruamel.yaml==0.16.10', 'colorama==0.4.4',
    'psutil==5.7.3', 'schema==0.7.3', 'scipy==1.9.1'
]

for pack in install_dependencies:
    cmd = 'pip install ' + pack
    os.system(cmd)

setup(
    name="rebyval",
    version="0.1",
    packages=find_packages(),
    author="Enzhi Zhang && Ruqin Wang",
    author_email="zhangenzhi657@gmail.com",
    description="A way to bridge regularization penalty with validation loss by a deep approximator",
    license="MIT",
    entry_points={
        'console_scripts':
            ['recmd = rebyval.tools.recmd.recmd:parse_args']
    },
    url="",
)
