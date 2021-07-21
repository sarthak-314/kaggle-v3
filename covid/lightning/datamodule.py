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
            'img': torch.tensor(img, dtype=torch.float), 
        }
        
        # add the target to the feature dict to make output dict
        output_dict = feature_dict
        target = row.label 
        output_dict['target'] = torch.tensor(target, dtype=torch.long)
        return output_dict
    
    def __len__(self): 
        return len(self.df)


class CompDatasetTest(torch.utils.data.Dataset): 
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform # ? TTA?
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        
        # read the main input 
        img = read_input_file(row.file_path)
        img = self.transform(img)
        
        # build all the features for the input
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float), 
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

# TODO: Add create_loader and other timm goodness to the dataloaders, datasets
class CompDataModule(pl.LightningDataModule):
    def __init__(self, dataframes, transforms, batch_size=32, num_workers):
        """
        :param dataframes (dict): processed train, valid and test dataframes
        :param transforms (dict): dict with transform function for train, valid, test
        """
        super().__init__()
        self.dataframes = dataframes
        self.transforms = transforms 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 4

    def train_dataloader(self):
        train_df = self.dataframes['train']
        train_transform = self.transforms['train']
        train_ds = CompDatasetTrain(train_df, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, 
            num_workers = self.num_workers, # you should optimize this 
            pin_memory = torch.cuda.is_available(), 
            shuffle = True, drop_last = True, 
        )
        return train_loader

    def val_dataloader(self):
        valid_df = self.dataframes['valid']
        valid_transform = self.transforms['valid']
        valid_ds = CompDatasetTrain(valid_df, valid_transform)
        val_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=self.batch_size, 
            num_workers = self.num_workers, 
            pin_memory = torch.cuda.is_available(), 
            shuffle = True, drop_last = False, 
        )
        return val_loader

    def test_dataloader(self): 
        test_df = self.dataframes['test']
        test_ds = CompDatasetTest(test_df)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=torch.cuda.is_available(), 
            shuffle=False, drop_last=False,
        )
        return test_loader

transforms = {
    'train': lambda train: train, 
    'valid': lambda valid: valid, 
    'test': lambda test: test, 
}