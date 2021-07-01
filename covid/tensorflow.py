import tensorflow as tf


# TENSORFLOW DATA FOR MAIN DATASET 
def decode_fn(path, img_size): 
    ext = 'png'
    file_bytes = tf.io.read_file(path)
    if ext == 'png':
        img = tf.image.decode_png(file_bytes, channels=3)
    elif ext in ['jpg', 'jpeg']:
        img = tf.image.decode_jpeg(file_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, (img_size, img_size))
    return img

def train_augment_fn(img):
    # TODO: Look More Into Augmentations 
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

def build_dataset(paths, labels, augment_fn=None, batch_size=32): 
    AUTO = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    img_ds = path_ds.map(decode_fn, num_parallel_calls=AUTO)
    if augment_fn is not None: 
        img_ds = img_ds.map(augment_fn, num_parallel_calls=AUTO)
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    ds = ds.repeat().shuffle(1024).batch(batch_size).prefetch(AUTO)
    return ds



