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
from chai.tensorflow.callbacks import * 
from chai.tensorflow.tfrecord import * 
from chai.tensorflow.layers import *
from chai.tensorflow.model import *
from chai.tensorflow.data import * 
from chai.tensorflow.lr import *

# Working Models & Datasets Imports
from chai.tensorflow.models.backbone_effnetv2 import download_effnetv2

# Function / Classes Imports
from chai.tensorflow.model import save_model

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

def enable_jit(): 
    tf.config.optimizer.set_jit(True)


def tf_lr_scheduler_factory(lr_scheduler_kwargs): 
    if isinstance(lr_scheduler_kwargs, int): 
        print(colored('Using constant learning rate', 'yellow'))
    lr_scheduler = tfa.optimizers.ExponentialCyclicalLearningRate(
        initial_learning_rate=1e-8, 
        maximal_learning_rate=lr_scheduler_kwargs['lr'], 
        step_size=lr_scheduler_kwargs['step_size'], 
        gamma=lr_scheduler_kwargs['gamma'], 
    )
    return lr_scheduler


def tf_optimizer_factory(optimizer_kwargs, lr_scheduler): 
    optimizer_name = optimizer_kwargs['name']
    if optimizer_name == 'AdamW': 
        optimizer = tfa.optimizers.AdamW(
            weight_decay=optimizer_kwargs['weight_decay'],
            learning_rate=lr_scheduler,  
            amsgrad=optimizer_kwargs['use_amsgrad'], 
        )
    if optimizer_kwargs['use_ranger']: 
        print(colored('Using Lookahead', 'red'))
        optimizer = tfa.optimizers.LookAhead(optimizer)
    if optimizer_kwargs['use_swa']: 
        print(colored('Using SWA', 'red'))
        optimizer = tfa.optimizers.SWA(optimizer)
    return optimizer


WORKING_DIR = Path('/content/')
TB_DIR = WORKING_DIR / 'tb-logs'