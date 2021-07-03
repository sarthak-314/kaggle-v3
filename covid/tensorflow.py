import tensorflow as tf
from utils.tensorflow import (
    decode_fn, augment_fn, resize_img, 
)
AUTO = tf.data.experimental.AUTOTUNE

def get_extension(path): 
    if path.endswith('png'): 
        return 'png'
    elif path.endswith('jpg'): 
        return 'jpg'

def comp_augment_fn(img): 
    return augment_fn(img)

def build_comp_dataset(paths, labels, img_size=256, augment_fn=None, batch_size=32):
    ext = get_extension(str(paths[0]))
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    full_img_ds = path_ds.map(lambda path: decode_fn(path, ext), num_parallel_calls=AUTO)
    # Cache the full sized images (not sure the best  way)
    full_img_ds = full_img_ds.cache()
    img_ds = full_img_ds.map(lambda img: resize_img(img, img_size), num_parallel_calls=AUTO)
    if augment_fn is not None: 
        img_ds = img_ds.map(augment_fn, num_parallel_calls=AUTO)
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    ds = ds.repeat().shuffle(1024).batch(batch_size).prefetch(AUTO)
    return ds
