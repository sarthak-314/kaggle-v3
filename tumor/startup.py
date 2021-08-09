# Some Setup to import utils
import sys 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 
from kaggle_utils.utils import solve_environment

# Competition Specific Constants
COMP_NAME = 'rsna-miccai-brain-tumor-radiogenomic-classification'
DRIVE_DIR = Path('/content/drive/MyDrive/Tumor')

# Competition Data 
MRI_TYPES = ['FLAIR','T1w','T1wCE','T2w']