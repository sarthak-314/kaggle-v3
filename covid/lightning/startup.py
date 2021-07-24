from datetime import timedelta
from torch import nn
import torchmetrics
import torch 
import timm

from covid.lightning.dataset import KaggleDatasetTrain, KaggleDatasetTest
from covid.lightning.datamodule import KaggleDataModule
from covid.lightning.callbacks import *
from covid.lightning.aug_timm import *


# Model Imports 
from torch.optim.lr_scheduler import CosineAnnealingLR
from covid.lightning.model import init_weights, load_backbone
from timm.optim.optim_factory import create_optimizer_v2
from torch.optim import AdamW