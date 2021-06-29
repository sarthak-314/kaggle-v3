"""
Common Startup Script for Notebooks
"""
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path 
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import subprocess
import warnings
import random
import shutil
import pickle
import torch
import json
import math
import glob
import cv2
import sys
import gc
import os

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output 
from IPython import get_ipython

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
ipython = get_ipython()
try: 
    ipython.magic('matplotlib inline')
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
except: 
    print('could not load ipython magic extensions')

# Install Libraries
subprocess.run(['pip', 'install' ,'awscli'])
subprocess.run(['pip', 'install' ,'wandb'])
import wandb

# Login to AWS & WandB
# TODO: Make a bash script and add sensitive information in .env file
subprocess.run(['aws', 'configure', 'set', 'aws_access_key_id', 'AKIAUD2OWH3DRHZEXSQE'])
subprocess.run(['aws', 'configure', 'set', 'aws_secret_access_key', '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'])
subprocess.run(['aws', 'configure', 'set', 'default.region', 'us-east-1'])
subprocess.run(['wandb', 'login', '00dfbb85e215726cccc6f6f9ed40e618a7cf6539'])


# Import all common functions / classes 
from utils.common import * 