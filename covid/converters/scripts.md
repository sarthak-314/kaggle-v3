# Basic Setup 
``` python 
# Working Dir + Tmp Dir
from pathlib import Path
WORKING_DIR = Path('/kaggle/working')
TMP_DIR = Path('/kaggle/tmp')

# Sync Notebook with VS Code
!git clone 'https://github.com/sarthak-314/kaggle-v3' {WORKING_DIR/'temp'}
%cd {WORKING_DIR/'temp'}
!git pull 
%run {WORKING_DIR/'temp/covid/startup.py'}

# Imports for the Notebook
import tensorflow_addons as tfa
import tensorflow_hub as hub
from utils.tensorflow import * 
from covid.tensorflow import * 
from covid.tensorflow_callbacks import *
STRATEGY = auto_select_accelerator()

# Import Competition Dataframes
train, valid, test = read_dataframes(FOLD)
```

# Download and Extract Zipped File from Drive
``` python
%%time 
# (13 minutes) Download and Extract Zipped Files
!pip install -q gdown

from pathlib import Path
import os

FILE_ID = '16Hy8E4Jt9G7XcbRWG19-X5xyWdjX4ma6'
DOWNLOAD_DIR = Path('/kaggle/tmp/download')
EXTRACT_DIR = Path('/kaggle/tmp/extracted')

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
!gdown {'https://drive.google.com/uc?id='+FILE_ID} -O {DOWNLOAD_DIR/'temp.zip'}
!unzip {DOWNLOAD_DIR/'temp.zip'} -d {EXTRACT_DIR}
```


# Create a Kaggle Dataset from a Folder
``` python
# Setup Kaggle Json file
!mkdir -p ~/.kaggle
!cp '/kaggle/input/kagglejson/kaggle.json' ~/.kaggle/
!cat ~/.kaggle/kaggle.json 
!chmod 600 ~/.kaggle/kaggle.json
import json

# Add metadata and create the dataset
DATASET_DIR = EXTRACT_DIR
dataset_metadata = {
    'title': 'Full PNG Train', 
    'id': 'readoc/full-png-train', 
    'licenses': [{'name': 'CC0-1.0'}]
}
with open(EXTRACT_DIR / 'dataset-metadata.json', 'w') as f:
    json.dump(dataset_metadata, f)
    
!kaggle datasets create -p {DATASET_DIR}
```

