import tensorflow as tf
try: 
    import tensorflow_addons as tfa
except Exception as e:
    print(e) 
try: 
    import tensorflow_hub as hub
except Exception as e: 
    print(e)

# Full Module Imports 
from tumor.tensorflow.callbacks import * 
from tumor.tensorflow.tfrecord import * 
from tumor.tensorflow.layers import *
from tumor.tensorflow.model import *
from tumor.tensorflow.data import * 
from tumor.tensorflow.lr import *

# Working Models & Datasets Imports
from tumor.tensorflow.models.backbone_effnetv2 import download_effnetv2

# Function / Classes Imports
from tumor.tensorflow.model import save_model

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)

def get_ranger(lr, min_lr=0): 
    radam = tfa.optimizers.RectifiedAdam( 
        learning_rate = lr,
        min_lr=min_lr,
        weight_decay=1e-4, # default is 0
        amsgrad = True,
        name = 'Ranger',
        #clipnorm = 10, # Not present by befault
    )
    return radam

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

def decode_fn(path, ext): 
    file_bytes = tf.io.read_file(path)
    if ext == 'png':
        img = tf.image.decode_png(file_bytes, channels=3)
    elif ext in ['jpg', 'jpeg']:
        img = tf.image.decode_jpeg(file_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def resize_img(img, img_size): 
    img = tf.image.resize(img, (img_size, img_size))
    return img

def augment_fn(img):
    # TODO: Look More Into Augmentations 
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

def enable_mixed_precision(): 
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Mixed precision enabled')

def enable_jit(): 
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')