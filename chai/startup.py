import sys

from sklearn.manifold import trustworthiness 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 
from kaggle_utils.utils import solve_environment
from chai.tensorflow_qa import *

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')

INTERNET_AVAILIBLE = True 
try: 
    os.system('pip install wandb')
    import wandb
except: 
    INTERNET_AVAILIBLE = False
    
    
# Termcolor Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')