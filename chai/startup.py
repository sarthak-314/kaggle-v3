import sys 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 
from kaggle_utils.utils import solve_environment
from chai.tensorflow_qa import *

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')
NUM_FOLDS = 5