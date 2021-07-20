from covid.lightning.datasets import CompDatasetTrain, CompDatasetTest
import pytorch_lightning as pl
import torch 


class CompDataModule(pl.LightningDataModule):
    def __init__(self, dataframes, transforms, batch_size=32, num_workers=8):
        """
        :param dataframes (dict): processed train, valid and test dataframes
        :param transforms (dict): dict with transform function for train, valid, test
        """
        super().__init__()
        self.dataframes = dataframes
        self.transforms = transforms 
        self.batch_size = batch_size
        self.num_workers = num_workers

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

"""
⚒ Ideas & Improvements ⚒
-------------------------
- add dataframes metadata like num_classes as @property 
- 
"""



