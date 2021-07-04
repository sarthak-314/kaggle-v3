from pathlib import Path
import pandas as pd
import numpy as np
import glob 
import ast
import os

import utils.dataframes
import covid

DATASET_NAME = 'siim-covid19-detection'
RAW_DATA_DIR = Path('/kaggle/input/siim-covid19-detection')
OUTPUT_DIR = Path('/kaggle/working/dataframes')

# CONFIG FOR BUILDING
NUM_FOLDS = 4
SPLIT_BY = 'group'
RANDOM_STATE = 42

# DATA FRAMES META DATA
LABEL_COLS = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']
CAPTIAL_TO_SMALL_STUDY_LABEL = {
    'Negative for Pneumonia': 'negative', 
    'Typical Appearance': 'typical', 
    'Indeterminate Appearance': 'intermediate', 
    'Atypical Appearance': 'atypical', 
}
SMALL_TO_CAPITAL_STUDY_LABEL = {v:k for k, v in CAPTIAL_TO_SMALL_STUDY_LABEL.items()}
DICOM_META_COLS = [
    'SOPInstanceUID', 'fname', 'Rows', 'Columns',  # Definately Useful Columns
    'BodyPartExamined', 'PatientSex', 'StudyDate', # Probably Useful 
    'ImageType', 'StudyTime', 'Modality', 'ImagerPixelSpacing', 'BitsAllocated', 'BitsStored', 
    'HighBit', 'PixelRepresentation', 'MultiImageType', # Might Be Useful
]


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

def read_raw_test(raw_data_dir):
    filepaths = glob.glob(str(raw_data_dir / 'test/**/*dcm'), recursive=True)
    test = pd.DataFrame({ 'img_path': filepaths })
    test['img_id'] = test.img_path.map(lambda x: _get_path_components(x)[-1].replace('.dcm', ''))
    test['study_id'] = test.img_path.map(lambda x: _get_path_components(x)[-3].replace('.dcm', ''))
    return test 

def read_raw_dataframes(raw_data_dir):
    # Read Raw Train
    train_study = pd.read_csv(raw_data_dir / 'train_study_level.csv')
    train_img = pd.read_csv(raw_data_dir / 'train_image_level.csv')
    train = _merge_input_dataframes(train_img, train_study)
    
    # Read Raw Test
    test = read_raw_test(raw_data_dir)
    
    return train, test

def standardize_train(train): 
    # One hot encode and add labels
    train['one_hot'] = train[LABEL_COLS].apply(lambda row: row.values, axis='columns')
    train['label'] = train.one_hot.apply(lambda array: np.argmax(array))

    # Add stratify column and group column
    train['stratify'] = train['one_hot'].apply(str)
    train['group'] = train['study_id'].apply(str)
    
    return train

"""
FEATURE ENGINEERING FUNCTIONS
-----------------------------
"""
def add_dicom_metadata(df, input_dir):
    from fastai.medical.imaging import get_dicom_files
    GET_PIXEL_SUMMARY = False
    dicom_files = get_dicom_files(input_dir)
    dicom_meta_df = pd.DataFrame.from_dicoms(dicom_files, px_summ=GET_PIXEL_SUMMARY)
    dicom_meta_df = dicom_meta_df[DICOM_META_COLS]
    dicom_meta_df = dicom_meta_df.rename(columns={
        'SOPInstanceUID': 'img_id', 
        'fname': 'dicom_img_path', 
        'Rows': 'img_height', 
        'Columns': 'img_width', 
    })
    df = df.merge(dicom_meta_df)
    return df

def post_process(df): 
    df.boxes = df.boxes.fillna('[]')
    df.boxes = df.boxes.apply(ast.literal_eval)
    return df

def build_folds(raw_data_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR): 
    train, _ = read_raw_dataframes(raw_data_dir)
    train = standardize_train(train)
    train = add_dicom_metadata(train, raw_data_dir/'train')
    train = post_process(train)
    fold_dfs = utils.dataframes.get_fold_dfs(train, SPLIT_BY, NUM_FOLDS)
    utils.dataframes.save_folds(fold_dfs, output_dir)
    return fold_dfs
    
def build_test(output_dir, raw_data_dir):
    _, test = read_raw_dataframes(raw_data_dir)
    test = add_dicom_metadata(test, raw_data_dir/'test')
    test = post_process(test)
    test.to_pickle(output_dir/'test.pkl')
    return test

    
DATAFRAMES_DIR = Path(os.path.dirname(covid.__file__)) / 'dataframes'
def read_dataframes(fold=0, dataframes_dir=DATAFRAMES_DIR):
    print('DATAFRAMES_DIR: ', dataframes_dir)
    os.listdir(dataframes_dir)
    train, valid = utils.dataframes.read_fold(fold, dataframes_dir, num_folds=NUM_FOLDS)
    test = pd.read_pickle(dataframes_dir/'test.pkl')
    return train, valid, test    