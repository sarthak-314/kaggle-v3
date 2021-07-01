"""
1. Build initial dataframes and upload to S3 
2. Read the dataframes and apply feature engineering
"""

from pathlib import Path
import pandas as pd
import numpy as np
import glob 
import os

import utils.dataframes

DATASET_NAME = 'siim-covid19-detection'
RAW_DATA_PATH = Path('/kaggle/input/siim-covid19-detection')

# Config for Building the Dataframes
NUM_FOLDS = 4
SPLIT_BY = 'group'
RANDOM_STATE = 42
HOLDOUT_PERCENTAGE = 1

# Additional Dataset-Specific Config
LABEL_COLS = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']


def _merge_input_dataframes(train_img, train_study):
    image_level_rename_map = { 'StudyInstanceUID': 'study_id', 'id': 'img_id' }
    train_img.id = train_img.id.str.replace('_image', '')
    train_img = train_img.rename(columns=image_level_rename_map)
    study_level_rename_map = {'id':'study_id'}
    train_study.id = train_study.id.str.replace('_study', '')
    train_study = train_study.rename(columns=study_level_rename_map)
    train = train_img.merge(train_study, on='study_id')
    return train

def _get_path_components(path): 
    normalized_path = os.path.normpath(path)
    path_components = normalized_path.split(os.sep)
    return path_components

def read_raw_test(raw_data_path):
    filepaths = glob.glob(str(raw_data_path / 'test/**/*dcm'), recursive=True)
    test = pd.DataFrame({ 'img_path': filepaths })
    test['img_id'] = test.img_path.map(lambda x: _get_path_components(x)[-1].replace('.dcm', ''))
    test['study_id'] = test.img_path.map(lambda x: _get_path_components(x)[-3].replace('.dcm', ''))
    return test 

def read_raw_dataframes(raw_data_path):
    # Read Raw Train
    train_study = pd.read_csv(raw_data_path / 'train_study_level.csv')
    train_img = pd.read_csv(raw_data_path / 'train_image_level.csv')
    train = _merge_input_dataframes(train_img, train_study)
    
    # Read Raw Test
    test = read_raw_test(raw_data_path)
    
    # Read Sample Submission
    sample_sub = pd.read_csv(raw_data_path / 'sample_submission.csv')
    
    return {
        'train': train, 
        'sample_sub': sample_sub, 
        'test': test
    }

def standardize_train(train): 
    # One hot encode and add labels
    train['one_hot'] = train[LABEL_COLS].apply(lambda row: row.values, axis='columns')
    train['label'] = train.one_hot.apply(lambda array: np.argmax(array))

    # Add stratify column and group column
    train['stratify'] = train['one_hot'].apply(str)
    train['group'] = train['study_id'].apply(str)
    
    return train

def build_and_save_folds(train, output_path=Path('/kaggle/working')): 
    fold_dfs = utils.dataframes.get_fold_dfs(df=train, split_by=SPLIT_BY, num_folds=NUM_FOLDS)
    utils.dataframes.save_folds(fold_dfs, output_path)
    print('Commit the notebook and then start feature engineering in next version')



"""
FEATURE ENGINEERING FUNCTIONS
-----------------------------
"""
# FastAI for .dicom files
try: 
    from fastai.basics import *
    from fastai.medical.imaging import *
except: 
    print('fastai import skipped')
    
DICOM_META_COLUMNS = [
    # Definatley Useful Columns
    'SOPInstanceUID', 'fname', 'Rows', 'Columns', 
    # Probably Useful 
    'BodyPartExamined', 'PatientSex', 'StudyDate',
    # Might Be Useful
    'ImageType', 'StudyTime', 'Modality', 'ImagerPixelSpacing', 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'MultiImageType',
]
COMP_INPUT_PATH = Path('/kaggle/input/siim-covid19-detection')
PX_SUMM = False
def add_dicom_metadata(df, df_type):
    dicom_files = get_dicom_files(COMP_INPUT_PATH / df_type)
    dicom_meta_df = pd.DataFrame.from_dicoms(dicom_files, px_summ=PX_SUMM) # 4 minutes for train without px_summ
    dicom_meta_df = dicom_meta_df[DICOM_META_COLUMNS]
    dicom_meta_df = dicom_meta_df.rename(columns={
        'SOPInstanceUID': 'img_id', 
        'fname': 'dicom_img_path', 
        'Rows': 'img_width', 
        'Columns': 'img_height', 
    })
    df = df.merge(dicom_meta_df)
    return df




# API FUNCTIONS
def preprocess_dataframes(raw_data_path, output_path): 
    raw_dataframes = read_raw_dataframes(raw_data_path)
    train, test = raw_dataframes['train'], raw_dataframes['test']
    train = standardize_train(train)
    build_and_save_folds(train, output_path=output_path)
    print('saving test')
    test.to_pickle(output_path/'test.pkl')

def read_dataframes(dataframes_path, fold=0):
    train, valid = utils.dataframes.read_fold(fold, dataframes_path, num_folds=NUM_FOLDS)
    return train, valid


def apply_feature_engineering_func(func, input_dataframes_path=Path(''), output_path=Path('/kaggle/working')):
    utils.dataframes.apply_feature_engineering_func(func, input_dataframes_path, output_path)
    
def build_test(raw_data_path): 
    test = read_raw_test(raw_data_path)
    # TODO: Apply all the feature engineering functions here
    return test