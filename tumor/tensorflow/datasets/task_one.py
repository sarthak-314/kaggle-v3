"""
Task 1 Tensorflow Dataset
"""
from termcolor import colored
import tensorflow as tf
import random 

from tumor.tensorflow.datasets.utils import *

def load_tfrecords(task1_gcs_paths, fold):
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
    
def _process_img(img, dim0, dim1, depth): 
    img = tf.cast(tf.io.decode_raw(img, tf.int16), tf.float32)
    img = tf.reshape(img, [dim0, dim1, depth]) # Reshape to an actual image
    return img

def read_tfrecord(example): 
    str_feat = tf.io.FixedLenFeature([], tf.string)
    int_feat = tf.io.FixedLenFeature([], tf.int64)
    string_features = { 'img': str_feat, 'segmentation': str_feat, 'patient_id': str_feat, 'mri_type': str_feat }
    int_features = { 'label_int': int_feat, 'dim0': int_feat, 'dim1': int_feat, 'depth': int_feat }
    features = { **string_features, **int_features }
    example = tf.io.parse_single_example(example, features)
    
    # Process Image & Segmentation
    dim0, dim1, depth = example['dim0'], example['dim1'], example['depth']
    img, seg = _process_img(example['img'], dim0, dim1, depth), _process_img(example['segmentation'], dim0, dim1, depth)
    
    return {
        'img': img,
        'segmentation': seg,
        'depth': depth, 
        'patient_id': example['patient_id'], 
        'mri_type': example['mri_type'], 
        'label_int': example['label_int'], 
    }


# Build & Cache the Datasets
def build_datasets_from_tfrecs(tfrecs):
    tfrec_ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)
    tfrec_ds = tfrec_ds.map(read_tfrecord, **AUTO).with_options(get_ignore_order()) 
    
    img_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['img'], **AUTO).cache()
    seg_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['segmentation'], **AUTO).cache()
    depth_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['depth'], **AUTO).cache()
    patient_id_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['patient_id'], **AUTO).cache()
    mri_type_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['mri_type'], **AUTO).cache()
    label_int_ds = tfrec_ds.map(lambda tfrec_out: tfrec_out['label_int'], **AUTO).cache()
    
    return {
        'img': img_ds,
        'segmentation': seg_ds,
        'depth': depth_ds,
        'patient_id': patient_id_ds,
        'mri_type': mri_type_ds,
        'label_int': label_int_ds,
    }
