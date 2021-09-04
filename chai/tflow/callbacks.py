from pathlib import Path
import tensorflow as tf 
import datetime
import os

MONITOR = 'val_loss'
MODE = 'min'
VERBOSE = 2

common_kwargs = {
    'monitor': MONITOR, 
    'mode': MODE, 
    'verbose': VERBOSE, 
}

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

def tb_callback(tb_dir, train_steps): 
    start_profile_batch = train_steps+10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"
    log_path = tb_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_path, histogram_freq=1, update_freq=20,
        profile_batch=profile_range, 
    )
    return tensorboard_callback

def model_checkpoint(checkpoint_dir=None):
    # checkpoint_filepath = 'checkpoint-{epoch:02d}-{val_loss:.4f}.h5'
    checkpoint_filepath = 'checkpoint.h5'
    if checkpoint_dir is not None: 
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = checkpoint_dir / checkpoint_filepath
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True, 
        **common_kwargs, 
    )

def early_stopping(patience=3):
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        restore_best_weights=True, 
        **common_kwargs,
    )

def reduce_lr_on_plateau(patience): 
    return tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=patience,
        min_delta=0.0001,
        min_lr=0,
        **common_kwargs, 
    )

def time_stopping(max_train_hours):
    import tensorflow_addons as tfa 
    return tfa.callbacks.TimeStopping(
        seconds=max_train_hours*3600
    )
    
def tqdm_bar(): 
    import tensorflow_addons as tfa
    return tfa.callbacks.TQDMProgressBar()

def terminate_on_nan(): 
    return tf.keras.callbacks.TerminateOnNaN()

def tensorboard_callback(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir)
    )

def wandb_callback():
    from wandb.keras import WandbCallback
    return WandbCallback()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )