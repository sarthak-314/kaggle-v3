from pytorch_lightning.callbacks import (
    BackboneFinetuning, EarlyStopping, LearningRateMonitor, ModelCheckpoint
)
from pl_bolts.callbacks import (
    TrainingDataMonitor, ModuleDataMonitor, BatchGradientVerificationCallback
)
from pl_bolts.callbacks.printing import PrintTableMetricsCallback
import os 

def get_backbone_finetuning(unfreeze_epoch=10, backbone_initial_ratio_lr=0.1, lr_multiplier=1.5): 
    # NOTE: Backbone should be at model.backbone
    
    # Scheduler for increasing backbone learning rate (epoch > unfreeze_epoch)
    multiplier = lambda epoch: lr_multiplier
    backbone_finetuning = BackboneFinetuning(
        unfreeze_backbone_at_epoch = unfreeze_epoch, # Backbone freezed till this epoch
        lambda_func = multiplier,
        backbone_initial_ratio_lr = backbone_initial_ratio_lr, # initial backbone lr = current lr * initial backbone ratio
        verbose = True, # Display the backbone and model learning rates
        should_align = True, # Align the learning rates
    )
    return backbone_finetuning

def get_early_stopping(patience=5, monitor='val/acc', mode='max'): 
    print(f'EarlyStopping: Will wait for {patience} epochs for the {monitor} to improve and then stop training')
    early_stopping = EarlyStopping(
        patience = patience, 
        monitor = monitor, 
        mode = mode, 
    )        
    return early_stopping

def get_lr_monitor(logging_interval='step'): 
    print(f'LearningRateMonitor: Will log learning rate for learning rate schedulers every {logging_interval} during training')
    return LearningRateMonitor(
        logging_interval = logging_interval, 
        log_momentum = True, # for moment based optimizers
    )

def get_model_checkpoint(checkpoint_dir='./', filename='epoch_{epoch:02d}-loss_{val/loss:.4f}-acc_{val/acc:.4f}', save_top_k=3, monitor='val/acc', mode='max'):
    checkpoint_dir = str(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Save top {save_top_k} models at {checkpoint_dir} with name {filename} if score increases')
    model_checkpoint = ModelCheckpoint(
        dirpath = checkpoint_dir,  
        filename = filename, 
        save_top_k = save_top_k, 
        save_last = True, 
        monitor = monitor, 
        mode = mode, 
    )
    return model_checkpoint

def get_print_table_metrics(): 
    print('PrintTableMetricsCallback: Will print table metrics at every epoch ending')
    return PrintTableMetricsCallback()

#TODO: Learn more about model pruning and it's callback
#TODO: Learn more about stochastic weight averaging and it's parameters

def get_training_data_monitor(log_every_n_steps=25): 
    print('Create histogram for each batch input in training step')
    return TrainingDataMonitor(
        log_every_n_steps = log_every_n_steps
    )
    
def get_module_data_monitor(): 
    print(f'ModuleDataMonitor: Histograms of data passing through a .forward pass')
    return ModuleDataMonitor()

def get_batch_grad_verification():     
    print(f'BatchGradientVerificationCallback: Lightweight verification callback')
    return BatchGradientVerificationCallback()




"""
from covid.lightning.callbacks import *

monitor = 'val/acc' 
mode = 'max'

unfreeze_epoch = 10 
backbone_initial_ratio_lr = 0.1 
lr_multiplier = 1.5 

checkpoint_dir = RUN_CHECKPOINT_DIR
filename = 'epoch_{epoch:02d}-loss_{val/loss:.4f}-acc_{val/acc:.4f}'
save_top_k = 5

early_stop_patience = 10

logging_interval = 'step'

callbacks = [
    get_backbone_finetuning(unfreeze_epoch=unfreeze_epoch, backbone_initial_ratio_lr=backbone_initial_ratio, lr_multiplier=lr_multiplier),
    get_model_checkpoint(checkpoint_dir=checkpoint_dir, filename=filename, save_top_k=save_top_k, monitor=monitor, mode=mode),
    get_early_stopping(patience=5, monitor=monitor, mode=mode),
    get_lr_monitor(logging_interval=logging_interval),
    get_print_table_metrics(),
    get_training_data_monitor(log_every_n_steps=25),
    get_module_data_monitor(),
    get_batch_grad_verification(),    
]
"""















