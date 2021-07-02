from utils.startup import * 
import ast

# Competition Specific
COMP_NAME = 'siim-covid19-detection'
COMP_INPUT_PATH = Path('/kaggle/input/siim-covid19-detection')

S3_PATH = 's3://siim-covid19-detection/'


def download_from_s3(dataset_name, output_path):
    output_path = str(output_path)
    bucket_path = S3_PATH + dataset_name
    subprocess.run(['aws', 's3', 'cp', bucket_path, output_path, '--recursive'], stdout=subprocess.PIPE)
    
# TODO: Build dataset_dicts for object detection
DATASETS = [
    'dataframes', # Dataframes for the competition
    'study_level_classification_full', # Full sized images for study level classification
    'image_level_classification_full', # Full sized images for image level classification
    'raddar_study_level_full', # Raddar's external dataset for study level classification
    'object_detection', # Object detection data from competition in coco format
]

import covid.datasets.dataframes
def read_dataframes(fold=0, tmp_folder=Path('./dataframes')): 
    tmp_folder = Path(tmp_folder)
    if not tmp_folder.exists(): 
        download_from_s3('dataframes', tmp_folder)
    train, valid = covid.datasets.dataframes.read_dataframes(
        tmp_folder, fold=fold
    )
    # Some Clearning
    if train.boxes.dtype == str: 
        train.boxes = train.boxes.fillna('[]')
        train.boxes = train.boxes.apply(ast.literal_eval)
    if valid.boxes.dtype == str: 
        valid.boxes = valid.boxes.fillna('[]')
        valid.boxes = valid.boxes.apply(ast.literal_eval)
    
    return train, valid

def get_all_filepaths(data_dir):
    filepaths = glob.glob(data_dir / '*' / '**', recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths