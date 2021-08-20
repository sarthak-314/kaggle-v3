# Some Setup to import utils
import sys 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 
from kaggle_utils.utils import solve_environment
from chai.tensorflow_qa import *

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')
NUM_FOLDS = 10

# Competition Data 
MRI_TYPES = ['FLAIR','T1w','T1wCE','T2w']

def read_fold(fold, dataframes_dir): 
    fold_dfs = []
    for fold_ in range(NUM_FOLDS): 
        fold_df = pd.read_pickle(dataframes_dir / f'fold{fold_}.pkl')
        fold_dfs.append(fold_df)
    train = pd.concat(fold_dfs[:fold]+fold_dfs[fold+1:])
    valid = fold_dfs[fold]
    return train, valid

DATAFRAMES_DIR = WORKING_DIR/'temp'/'tumor'/'dataframes'
def read_dataframes(fold, dataset_name=COMP_NAME): 
    train, valid = read_fold(fold, DATAFRAMES_DIR/dataset_name)
    return train, valid    


