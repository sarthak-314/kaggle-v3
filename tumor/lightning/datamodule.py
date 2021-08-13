from covid.lightning.dataset import KaggleDatasetTrain, KaggleDatasetTest
from covid.lightning.loader import create_loader
import pytorch_lightning as pl
class KaggleDataModule(pl.LightningDataModule): 
    def __init__(self, dataframes, transforms, batch_size=32):
        """
        :param dataframes: processed train & valid dataframes
        :param transforms: transforms for training and evaluation
        """
        super().__init__()
        self.train_df, self.valid_df = dataframes
        self.train_transforms, self.eval_transforms = transforms
        self.batch_size = batch_size

    def train_dataloader(self):
        train_ds = KaggleDatasetTrain(self.train_df, self.train_transforms)
        train_loader = create_loader(train_ds, self.batch_size, is_training=True)
        return train_loader

    def val_dataloader(self):
        valid_ds = KaggleDatasetTrain(self.valid_df, self.eval_transforms)
        val_loader = create_loader(valid_ds, self.batch_size, is_training=False)
        return val_loader
