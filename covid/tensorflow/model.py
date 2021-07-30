import tensorflow_addons as tfa
from termcolor import colored
import tensorflow_hub as hub
import tensorflow as tf
from time import time
import wandb
import os 

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def optimizer_factory(optimizer_args):
    optimizer_name, lr = optimizer_args['optimizer'], optimizer_args['lr']
    if optimizer_name.lower() == 'adam': 
        return tf.keras.optimizers.Adam(lr)
    elif optimizer_name.lower() == 'ranger': 
        radam = tfa.optimizers.RectifiedAdam( 
            lr=lr,
            total_steps=optimizer_args['total_steps'], 
            warmup_proportion=optimizer_args['warmup_proportion'], 
            amsgrad=optimizer_args['amsgrad'],
            min_lr=0,
            weight_decay=1e-4, 
            name='Ranger',
        )
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        return ranger


def compile_model(model, strategy, loss, metrics=['accuracy'], optimizer_args=None): 
    with strategy.scope(): 
        model.compile(
            loss=loss,
            metrics=metrics, 
            optimizer=optimizer_factory(optimizer_args), 
            steps_per_execution=64,
        )
    print('Model Compiled')


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(filepath=path, options=get_save_locally())
    model.save_weights(filepath=str(path/f'weights.h5'), options=get_save_locally())
    try: 
        wandb.save(str(path/'weights.h5'))
    except: 
        print('Skipping wandb save')

def save_weights(model, filepath):
    filepath = str(filepath)
    print('Saving model weights at', colored(filepath, 'blue'))
    model.save_weights(filepath=filepath, options=get_save_locally())
    