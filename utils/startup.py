from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path 
from tqdm import tqdm
from PIL import Image
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

# WandB Setup
subprocess.run(['pip', 'install' ,'wandb'])
subprocess.run(['wandb', 'login', '00dfbb85e215726cccc6f6f9ed40e618a7cf6539'])
import wandb


# Useful Functions 
def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '*' / '**'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths


# Logger
from urllib.parse import urlencode
import urllib.request as request
import threading
import sys
class Logger(object):
    CHANNEL_NAME = 'timm-pretraining-yash-v1'
    def __init__(self):
        self.terminal = sys.stdout
    def write(self, message):
        if message != '\n':
            self.terminal.write(message + '\n')
            payload = {'msg': message}
            quoted = urlencode(payload)
            thr = threading.Thread(target=self.send, args=(quoted,), kwargs={})
            thr.start()
    def flush(self):
        pass
    @staticmethod
    def send(msg):
        msg = 'https://dweet.io/dweet/for/' + Logger.CHANNEL_NAME + '?' + msg
        try:
            request.urlopen(msg).read()
        except Exception as e:
            sys.stdout.terminal.write(e)

def setup_logger(channel_name):
    Logger.CHANNEL_NAME = channel_name
    sys.stdout = Logger()
    print('Started')









