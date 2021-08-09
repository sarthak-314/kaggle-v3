
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm.auto import tqdm
from pathlib import Path 
from PIL import Image
from time import time
import pandas as pd
import numpy as np
import subprocess
import warnings
import random
import shutil
import pickle
try: 
    import torch
except: 
    print('Could not import torch')
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
    if 'KAGGLE_CONTAINER_NAME' in os.environ: 
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
        # Install Libraries
        os.system('pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib')
        os.system('pip install wandb')
        subprocess.run(['wandb', 'login', '00dfbb85e215726cccc6f6f9ed40e618a7cf6539'])
        import wandb
    except: 
        online = False
    return online    

ENV = solve_env()
HARDWARE, DEVICE = solve_hardware()
ONLINE = solve_internet_status()
print(f"Running on {colored(DEVICE, 'green')} on {ENV} with internet {['on', 'off'][ONLINE]}")

RANDOM_STATE = 69420

# Useful Paths depending on environment 
KAGGLE_INPUT_DIR = Path('/kaggle/input')
if ENV == 'Colab': 
    DRIVE_DIR = Path('/content/drive/MyDrive')
    WORKING_DIR = Path('/content')
    TMP_DIR = Path('/content/tmp')
elif ENV == 'Kaggle': 
    WORKING_DIR = Path('/kaggle/working')
    TMP_DIR = Path('/kaggle/tmp')
    

# Sync this Code with Jupyter Notebook via Github
MY_CODE_REPO = 'https://github.com/sarthak-314/kaggle-v3'
def sync(): 
    os.chdir(WORKING_DIR/'temp')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'temp'))
    
# Clone a GitHub Repo
def clone_repo(repo_url): 
    repo_name = repo_url.split('/')[-1].replace('-', '_')
    clone_dir = str(WORKING_DIR/repo_name)
    subprocess.run(['git', 'clone', repo_url, clone_dir])
    os.chdir(clone_dir)
    sys.path.append(clone_dir)
    print(f'Repo {repo_url} cloned')    

    
# Log Remotely to 
from urllib.parse import urlencode
import urllib.request as request
import threading 
class Logger(object):
    CHANNEL_NAME = 'my-channel'
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

def log_remotely(channel_name): 
    print(f'Logging output to https://shantanum91.github.io/kagglewatch/ on channel {channel_name}')
    Logger.CHANNEL_NAME = channel_name
    sys.stdout = Logger()
    
    
def get_gcs_path_fn(gcs_path, dataset_dir): 
    def fn(org_path, dataset_dir=dataset_dir ,gcs_path=gcs_path):
        org_path, dataset_dir = str(org_path), str(dataset_dir)
        gcs_path = org_path.replace(dataset_dir, gcs_path)
        return gcs_path
    return fn

def oversample(train, class_to_oversample_ratio): 
    print(train.label.value_counts())
    org=len(train)
    print('Original Training Samples: ', org)
    print('Class weights: ', compute_class_weight('balanced', list(range(train.label.nunique())), train.label.values))
    oversampled_labels = []
    for label, times in class_to_oversample_ratio.items():
        label_df = train[train.label==label]
        oversampled_label = pd.concat([label_df]*times)
        oversampled_labels.append(oversampled_label)
    df = pd.concat(oversampled_labels).sample(frac=1)
    new=len(train); new; new/org
    print('New train samples: ', new)
    print('new/original: ', new/org)
    print(train.label.value_counts())
    print('Class weights: ', compute_class_weight('balanced', list(range(train.label.nunique())), train.label.values))
    return df


def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '**' / '*'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths

def get_img_path_fn(filepaths): 
    def get_img_path(img_id): 
        for fp in filepaths: 
            if img_id in fp: 
                return fp
        print(f'img id {img_id} not found in filepaths')
    return get_img_path


# Check if repo is loaded correctly
if ENV == 'Kaggle': 
    assert Path('/kaggle/working/temp').exists() 
elif ENV == 'Colab': 
    assert Path('/content/temp').exists()
    
# Mount Drive
def mount_drive(): 
    from google.colab import drive
    drive.mount('/content/drive')