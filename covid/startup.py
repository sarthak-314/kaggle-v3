from utils.startup import * 
from utils.utils import solve_environment

from covid.dataframes import (
    LABEL_COLS, CAPTIAL_TO_SMALL_STUDY_LABEL, SMALL_TO_CAPITAL_STUDY_LABEL, DICOM_META_COLS, \
    read_dataframes, 
)

# Competition Specific
COMP_NAME = 'siim-covid19-detection'
COMP_DIR = Path('/kaggle/input/siim-covid19-detection')

COMP_DRIVE_DIR = Path('/content/drive/MyDrive/Covid')
COMP_DRIVE_DATA_DIR = Path('/content/drive/MyDrive/Covid/Data')
DRIVE_DATAFRAMES_DIR = COMP_DRIVE_DATA_DIR / 'Dataframes'

def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '*' / '**'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths

def read_xray(path, voi_lut = False, fix_monochrome = True):
    import pydicom
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

