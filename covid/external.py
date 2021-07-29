import pandas as pd
import glob

def get_all_filepaths(data_dir):
    filepaths = glob.glob(str(data_dir / '**' / '*'), recursive=True) 
    print(f'{len(filepaths)} files found in {data_dir}')    
    return filepaths

def build_cxr(dataset_dir): 
    cxr_df = pd.read_csv(dataset_dir/'train.txt', sep=' ', header=None)
    cxr_df.columns = ['patient_id', 'filename', 'label', 'source']
    cxr_df = cxr_df[cxr_df.filename.str.contains('png')]
    cxr_df = cxr_df[cxr_df.source!='actmed'] # BUG: BMP Images
    cxr_df['img_path'] = cxr_df.filename.apply(lambda fp: str(dataset_dir/'train'/fp))
    return cxr_df

def build_bimcv(dataset_dir): 
    filepaths = get_all_filepaths(dataset_dir)
    df_dict = {
        'img_path': filepaths, 
        'label': ['covid']*len(filepaths)
    }
    df = pd.DataFrame.from_dict(df_dict)
    return df

def build_covid19_radiography(dataset_dir): 
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
    return df

def build_chest_xray_pneumonia(dataset_dir): 
    filepaths = get_all_filepaths(dataset_dir)
    df_dict = {'img_path': [], 'label': []}
    for filepath in filepaths: 
        if 'PNEUMONIA' in filepath: 
            label = 'pneumonia'
        elif 'NORMAL' in filepath: 
            label = 'normal'
        df_dict['img_path'].append(filepath)
        df_dict['label'].append(label)
    df = pd.DataFrame.from_dict(df_dict)
    return df

