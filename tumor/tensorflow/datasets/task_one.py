"""
Task 1 Tensorflow Dataset
"""
from termcolor import colored
import tensorflow as tf
import random 

def load_task1_tfrecords(task1_gcs_paths, fold):
    train_tfrecs, valid_tfrecs = [], []
    for fold_, fold_gcs_path in enumerate(task1_gcs_paths): 
        if fold_ == fold: 
            valid_tfrecs = tf.io.gfile.glob(f'{fold_gcs_path}/*tfrec')
        else: 
            train_tfrecs += tf.io.gfile.glob(f'{fold_gcs_path}/*tfrec')
    random.shuffle(train_tfrecs); random.shuffle(valid_tfrecs)
    print(f"{colored(len(train_tfrecs), 'blue')} train tfrecords found")
    print(f"{colored(len(valid_tfrecs), 'blue')} valid tfrecords found")
    return train_tfrecs, valid_tfrecs
    
    