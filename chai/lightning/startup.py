from datetime import timedelta
from torch import nn
import torchmetrics
import torch 

from tumor.lightning.dataset import KaggleDatasetBase
from tumor.lightning.datamodule import KaggleDataModule, get_data_module_class
from tumor.lightning.callbacks import *
from tumor.lightning.aug_timm import *
from tumor.lightning.loader import *


# Model Imports 
from torch.optim.lr_scheduler import CosineAnnealingLR
# from covid.lightning.model import init_weights, load_timm_backbone
# from timm.optim.optim_factory import create_optimizer_v2
from torch.optim import AdamW