class Demand:
    def __init__(self,
                 graph,
                 num_demands,
                 job_centric=False):
        pass



'''
pip install has extra stuff to do but worth

README.md
LICENSE.md
setup.py




python setup.py develop (when have done) (use for making importable from anywhere on your PC)
inside setup.py:
import setuptools
setuptools.setup(
    name="gcn",
    version="0.0.1",
    author="Zak Shabka",
    author_email="zak@shabka.com",
    description="Graph Convolutional Network implemented with DGL",
    url="https://github.com/zawaki/dgl_gcn",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)





inside top level __init__.py (to make all sub module functions callable from top level):
from gcn.aggregators import *
from gcn.model import *
from gcn.supervised_train import *
from gcn.unsupervised_train import *
from gcn.tensorboard_writer import *


'''
