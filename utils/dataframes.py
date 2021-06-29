"""
Utility functions for processing, and feature engineering on datasets
"""
# Library Imports
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split
from distutils.dir_util import copy_tree
from pathlib import Path
from time import time
from tqdm import tqdm
import pandas as pd
import glob
import math
import os 

def get_fold_dfs(df, split_by, num_folds): 
    fold_class = {'group': GroupKFold, 'stratify': StratifiedKFold}[split_by]
    fold_fn = fold_class(num_folds)
    fold_dfs = []
    for fold_, (_, val_idx) in enumerate(fold_fn.split(df, df, df[split_by])): 
        fold_df = df.iloc[val_idx]
        fold_dfs.append(fold_df)
    return fold_dfs

def save_folds(fold_dfs, output_path): 
    start_time = time()
    os.makedirs(output_path, exist_ok=True)
    for fold, fold_df in enumerate(fold_dfs):
        df_path = str(output_path / f'fold_{fold}')
        fold_df.to_pickle(df_path+'.pkl')
        print(f'{fold} saved')
    print(f'{time() - start_time} to build {len(fold_dfs)} folds')    

def read_fold(fold, input_dataframes_path, num_folds):  
    fold_dfs = []
    for fold_ in range(num_folds): 
        fold_df = pd.read_pickle(input_dataframes_path / f'fold_{fold_}.pkl')
        fold_dfs.append(fold_df)

    train = pd.concat(fold_dfs[:fold]+fold_dfs[fold+1:])
    valid = fold_dfs[fold]
    return train, valid


def feature_col(func):
    def wrapper(df):
        def func_wrapper(row): 
            kwargs = {}
            for col in func.__code__.co_varnames: 
                kwargs[col] = row[col]
            return func(**kwargs)
        df[func.__name__] = df.apply(func_wrapper, axis='columns')
        return df
    return wrapper 

def apply_feature_engineering_func(func, input_dataframes_path, output_path): 
    start_time = time()
    # Copy everything from input_folder to output_folder
    if input_dataframes_path != output_path: 
        copy_tree(str(input_dataframes_path), str(output_path))
        print(f'reading dataframes from {input_dataframes_path} and writing to {output_path}')
    else: 
        print('reading from and writing to the same folder')    
    
    # Read all the dataframe related files
    all_pkl_files = glob.glob(str(output_path/'**/*.pkl'), recursive=True)
    print('all files read')
    
    for df_file in tqdm(all_pkl_files): 
        df = pd.read_pickle(df_file)
        df_type = 'test' if 'test' in df_file else 'train'
        try: 
            df = func(df, df_type=df_type)
            df.to_pickle(df_file)
        except: 
            print(f'{func.__name__} did not work on {df_file}. Skipping applying it')
    print(f'{time()-start_time} seconds to build the features') #23 secs
