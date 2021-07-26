# Some Setup to import utils
import sys 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')


from utils.startup import * 
from tqdm.auto import tqdm # Lazy
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
ZIPPED_DIR = COMP_DRIVE_DATA_DIR / 'Zipped'

def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '**' / '*'), recursive=True) 
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

def get_img_path_fn(filepaths): 
    def get_img_path(img_id): 
        for fp in filepaths: 
            if img_id in fp: 
                return fp
        print(f'img id {img_id} not found in filepaths')
    return get_img_path

def add_kaggle_and_gcs_path(train, valid, kaggle_dataset): 
    dataset_dir = KAGGLE_INPUT_DIR/kaggle_dataset
    filepaths = get_all_filepaths(dataset_dir)
    get_img_path = get_img_path_fn(filepaths)
    train['kaggle_path'] = train.img_id.apply(get_img_path)
    valid['kaggle_path'] = valid.img_id.apply(get_img_path)
    try: 
        from kaggle_datasets import KaggleDatasets
        gcs_path = KaggleDatasets().get_gcs_path(kaggle_dataset)
        get_gcs_path = get_gcs_path_fn(gcs_path, dataset_dir)
        train['gcs_path'] = train.img_path.apply(get_gcs_path)
        valid['gcs_path'] = valid.img_path.apply(get_gcs_path)
    except: 
        print('Could not add gcs path')
    return train, valid