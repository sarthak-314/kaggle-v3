from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow as tf

# Full Imports 
from covid.tensorflow.model import * 
from covid.tensorflow.data import * 

def auto_select_accelerator():
    """
    Auto Select TPU / GPU / CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def get_gcs_path(dataset_name): 
    # Shortcuts for Colab 
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
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

# Tensorflow Setup in Jupyter
STRATEGY = auto_select_accelerator()
enable_mixed_precision()
