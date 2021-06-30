from wandb.keras import WandbCallback
import tensorflow_addons as tfa
import tensorflow as tf 

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
    return tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
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
    return tfa.callbacks.TerminateOnNaN()

def tensorboard(log_dir):
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir)
    )

def wandb():
    return WandbCallback()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )
    
