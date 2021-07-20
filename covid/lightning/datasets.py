"""
Main module for creating torch Datasets from processed dataframes
"""

import torch 
from src.data.comp.read_dataset import (
    FEATURE_COLS, TARGET_COL, 
    read_dataframes, read_input_file
)
from src.aug.apply_transform import train_transform_comp

class CompDatasetTrain(torch.utils.data.Dataset): 
    def __init__(self, df, transform):
        """
        Build torch Dataset for the train/valid dataframe 
        
        Args:
            df (DataFrame): train or valid dataframe
            transform (function): function to apply to input file to augment it

        Returns:
            output_dict (dict): output dictionary for each input containing features and target
        """ 
        self.df = df
        self.transform = transform
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        
        # read the main input 
        img = read_input_file(row.file_path)
        img = self.transform(img)
        
        # build all the features for the input
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float)
        }
        
        # add the target to the feature dict to make output dict
        output_dict = feature_dict
        target = row.label 
        output_dict['target'] = torch.tensor(target, dtype=torch.long)
        return output_dict
    
    def __len__(self): 
        return len(self.df)


class CompDatasetTest(torch.utils.data.Dataset): 
    def __init__(self, df):
        self.df = df
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        
        # read the main input 
        img = read_input_file(row.file_path)
        img = self.transform(img)
        
        # build all the features for the input
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float)
        }
        # no labels for test
        output_dict = feature_dict
        return output_dict
    
    def __len__(self): 
        return len(self.df)    

# testing basic
train, test = read_dataframes()['train'], read_dataframes()['test']
train_ds = CompDatasetTrain(train, train_transform_comp)
test_ds = CompDatasetTest(test)