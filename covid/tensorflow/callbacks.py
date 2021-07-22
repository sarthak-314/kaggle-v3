from utils.tensorflow import get_load_locally
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
import tensorflow as tf 
import os

from utils.tensorflow import get_save_locally

# Config for the Competition
MONITOR = 'val_acc'
MODE = 'max'
VERBOSE = 2

common_kwargs = {
    'monitor': MONITOR, 
    'mode': MODE, 
    'verbose': VERBOSE, 
}

def get_model_checkpoint(checkpoint_path): 
    os.makedirs(checkpoint_path, exist_ok=True)
    return tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        options=get_save_locally(), 
        **common_kwargs, 
    )
    
def get_early_stopping(patience=3):
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        restore_best_weights=True, 
        **common_kwargs,
    )
    
def get_reduce_lr_on_plateau(): 
    return tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=5,
        min_delta=0.0001,
        min_lr=0,
        **common_kwargs, 
    )

def time_stopping(max_train_hours): 
    return tfa.callbacks.TimeStopping(
        seconds=max_train_hours*3600
    )
    
def tqdm_bar(): 
    return tfa.callbacks.TQDMProgressBar()

def terminate_on_nan(): 
    return tf.keras.callbacks.TerminateOnNaN()

def tensorboard_callback(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir)
    )

def wandb_callback():
    return WandbCallback()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )
    

def get_lr_callback(lr_min=1e-6, lr_warmup_epochs=10, lr_max=1e-3):
    def lrfn(epoch): 
        EXP_DECAY = 0.9
        if epoch < lr_warmup_epochs: 
            lr = (lr_max-lr_min) / lr_warmup_epochs * epoch + lr_min
        else: 
            lr = (lr_max-lr_min) * EXP_DECAY ** (epoch-lr_warmup_epochs) + lr_min
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    return lr_callback