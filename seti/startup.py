from utils.startup import * 
from utils.utils import solve_environment

from covid.dataframes import (
    LABEL_COLS, CAPTIAL_TO_SMALL_STUDY_LABEL, SMALL_TO_CAPITAL_STUDY_LABEL, DICOM_META_COLS, \
    read_dataframes, 
)

# Competition Specific
COMP_NAME = 'seti-breakthrough-listen'

COMP_DRIVE_DIR = Path('/content/drive/MyDrive/Seti')
COMP_DRIVE_DATA_DIR = Path('/content/drive/MyDrive/Seti/Data')