from pathlib import Path

import utils
import utils.utils


DATASET_NAME = 'google-smartphone-decimeter-challenge'
# Environment Dependent Paths
ENV = utils.utils.solve_environment()
if ENV == 'COLAB': 
    RAW_DATA_PATH = Path('/content/drive/MyDrive/Decimeter/Original')
else: 
    RAW_DATA_PATH = Path(f'/kaggle/input/{DATASET_NAME}')

# Config for Building Dataframes
NUM_FOLDS = 4
SPLIT_BY = 'group'
RANDOM_STATE = 42

# Dataframes Meta Data

    
