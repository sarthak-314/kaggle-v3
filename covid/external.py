from utils.startup import KAGGLE_INPUT_DIR
from sklearn.utils.class_weight import compute_class_weight
from termcolor import colored
from pathlib import Path
import pandas as pd
import glob
import os 

KAGGLE_INPUT_DIR = Path('/kaggle/input')

def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '**' / '*'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths

def post_process(df, kaggle_dataset):
    print('Processing', colored(kaggle_dataset, 'blue')) 
    print(f'Length of {kaggle_dataset} df: ', colored(len(df), 'blue'))
    df = df[['img_path', 'label']]
    df.img_path = df.img_path.apply(str)
    df['img_ext'] = df.img_path.apply(lambda x: x.split('.')[-1])
    df = df[(df.img_ext=='png') | (df.img_ext=='jpg') | (df.img_ext=='jpeg')]
    print(df.label.value_counts())
    print()
    return df

def build_cxr(dataset_dir=KAGGLE_INPUT_DIR/'covidx-cxr2'): 
    kaggle_dataset = 'covidx-cxr2'
    cxr_df = pd.read_csv(dataset_dir/'train.txt', sep=' ', header=None)
    cxr_df.columns = ['patient_id', 'filename', 'label', 'source']
    cxr_df = cxr_df[cxr_df.source!='actmed'] # BUG: BMP Images
    cxr_df['img_path'] = cxr_df.filename.apply(lambda fp: str(dataset_dir/'train'/fp))
    label_map = {'negative': 'normal', 'positive': 'covid'}
    cxr_df.label = cxr_df.label.map(label_map)
    return post_process(cxr_df, kaggle_dataset)

def build_bimcv(dataset_dir=KAGGLE_INPUT_DIR/'bimcv-external'): 
    kaggle_dataset = 'bimcv-external'
    filepaths = get_all_filepaths(dataset_dir)
    df_dict = {
        'img_path': filepaths, 
        'label': ['covid']*len(filepaths)
    }
    df = pd.DataFrame.from_dict(df_dict)
    return post_process(df, kaggle_dataset)

def build_covid19_radiography(dataset_dir=KAGGLE_INPUT_DIR/'covid19-radiography-database'): 
    kaggle_dataset = 'covid19-radiography-database'
    filepaths = get_all_filepaths(dataset_dir)
    df_dict = {'img_path': [], 'label': []}
    for filepath in filepaths: 
        if 'png' not in filepath: continue
        if 'Lung_Opacity' in filepath: continue
        if 'Viral Pneumonia' in filepath: 
            label = 'pneumonia'
        elif 'Normal' in filepath: 
            label = 'normal'
        elif 'COVID' in filepath: 
            label = 'covid'
        df_dict['img_path'].append(filepath)
        df_dict['label'].append(label)
    df = pd.DataFrame.from_dict(df_dict)
    return post_process(df, kaggle_dataset)

def build_chest_xray_pneumonia(dataset_dir=KAGGLE_INPUT_DIR/'chest-xray-pneumonia'): 
    kaggle_dataset = 'chest-xray-pneumonia'
    filepaths = get_all_filepaths(dataset_dir)
    df_dict = {'img_path': [], 'label': []}
    for filepath in filepaths: 
        if os.path.isdir(filepath): continue
        if 'PNEUMONIA' in filepath: 
            label = 'pneumonia'
        elif 'NORMAL' in filepath: 
            label = 'normal'
        df_dict['img_path'].append(filepath)
        df_dict['label'].append(label)
    df = pd.DataFrame.from_dict(df_dict)
    return post_process(df, kaggle_dataset)


def process_df(df, labels):
    if isinstance(df, list): 
        df = pd.concat(df)
    df = df[['img_path', 'label']]
    df.img_path = df.img_path.apply(str)
    df['img_ext'] = df.img_path.apply(lambda x: x.split('.')[-1])
    df = df[(df.img_ext=='png') | (df.img_ext=='jpg') | (df.img_ext=='jpeg')]
    label2idx = {label:idx for idx, label in enumerate(labels)}
    print(f'Label to index: {label2idx}')
    df.label = df.label.map(label2idx)
    print(df.label.value_counts())
    print('Length of df: ', colored(len(df), 'green'))
    return df
