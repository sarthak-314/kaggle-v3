import tensorflow as tf
import wandb
import os 

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(filepath=path, options=get_save_locally())
    model.save_weights(filepath=str(path/f'weights.h5'), options=get_save_locally())
    wandb.save(str(path/'weights.h5'))