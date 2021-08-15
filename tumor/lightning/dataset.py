from PIL import Image
import torch 

def read_img(img_path): 
    img = Image.open(img_path).convert('RGB')
    return img


class KaggleDatasetBase(torch.utils.Dataset): 
    """
    Base class to build a torch Dataset for train/valid
    Args:
        df (DataFrame): train or valid dataframe
        df should contain a 'label' column and an 'img_path' column
        transform (function): function to apply to image to augment it

    Returns:
        feature_dict (dict): dictionary of all the features required by the net
        output_dict (dict): outputs for the net to learn from
    """ 
    def __init__(self, df, transform): 
        self.df = df
        self.transform = transform
        print(f'Received a dataframe of length {len(df)} for dataset')
    
    def __len__(self): 
        return len(self.df)


class KaggleDatasetTrain(torch.utils.data.Dataset): 

    def __init__(self, df, transform): 
        self.df = df
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_img(row.img_path)
        img = self.transform(img)
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float), 
        }
        output_dict = {
            **feature_dict, 
            'target': torch.tensor(row.label, dtype=torch.long)
        }
        return output_dict 
    
    def __len__(self): 
        return len(self.df)
    
class KaggleDatasetTest(torch.utils.data.Dataset): 
    def __init__(self, df, transform): 
        self.df = df
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_img(row.img_path)
        img = self.transform(img)
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float), 
        }
        output_dict = {
            **feature_dict, 
        }
        return output_dict 
    
    def __len__(self): 
        return len(self.df)