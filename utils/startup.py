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


# Useful Functions 
def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '*' / '**'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths


# Solve Environment, Hardware & Online Status
def solve_env(): 
    if Path('/kaggle/').exists():
        return 'Kaggle'
    elif Path('/content/').exists(): 
        return 'Colab'
    
def solve_hardware(): 
    hardware = 'CPU'
    device_name = 'CPU'
    if torch.cuda.is_available(): 
        hardware = 'GPU'
        device_name = torch.cuda.get_device_name(0)
    elif 'TPU_NAME' in os.environ: 
        hardware = 'TPU'
        device_name = 'TPU v3'
    return hardware, device_name

def solve_internet_status(): 
    online = True
    try:  
        # WandB Setup
        subprocess.run(['pip', 'install' ,'wandb'])
        subprocess.run(['wandb', 'login', '00dfbb85e215726cccc6f6f9ed40e618a7cf6539'])
        import wandb
    except: 
        online = False
    return online    

ENV = solve_env()
HARDWARE, DEVICE = solve_hardware()
ONLINE = solve_internet_status()
print(f"Running on {colored(DEVICE, 'green')} on {ENV} with internet {['on', 'off'][ONLINE]}")

# Useful Paths depending on environment 
if ENV == 'Colab': 
    DRIVE_DIR = Path('/content/drive/MyDrive')
    WORKING_DIR = Path('/content')
    TMP_DIR = Path('/content/tmp')
elif ENV == 'Kaggle': 
    KAGGLE_INPUT_DIR = Path('/kaggle/input')
    WORKING_DIR = Path('/kaggle/working')
    TMP_DIR = Path('/kaggle/tmp')
    

# Sync this Code with Jupyter Notebook via Github
MY_CODE_REPO = 'https://github.com/sarthak-314/kaggle-v3'
def sync(): 
    os.chdir(WORKING_DIR/'temp')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'temp'))