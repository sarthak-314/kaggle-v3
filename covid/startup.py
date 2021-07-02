from utils.startup import * 

from covid.datasets.dataframes import (
    LABEL_COLS, CAPTIAL_TO_SMALL_STUDY_LABEL, SMALL_TO_CAPITAL_STUDY_LABEL, DICOM_META_COLS, \
    read_dataframes, 
)

# Competition Specific
COMP_NAME = 'siim-covid19-detection'
COMP_INPUT_PATH = Path('/kaggle/input/siim-covid19-detection')
S3_PATH = 's3://siim-covid19-detection/'

try: 
    dataframes_dir = Path('./dataframes')
    train, valid, test = read_dataframes(0, dataframes_dir)
except Exception as e: 
    print(f'Exception reading dataframes: {e}')

def download_from_s3(dataset_name, output_path):
    output_path = str(output_path)
    bucket_path = S3_PATH + dataset_name
    subprocess.run(['aws', 's3', 'cp', bucket_path, output_path, '--recursive', '--quiet'], stdout=subprocess.PIPE)

def upload_to_s3(folder_path, dataset_name):
    folder_path = str(folder_path)
    bucket = S3_PATH + dataset_name
    subprocess.run(['aws', 's3', 'cp', folder_path, bucket, '--recursive', '--quiet'], stdout=subprocess.PIPE)

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

