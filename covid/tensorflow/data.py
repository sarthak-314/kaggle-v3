import tensorflow as tf

def get_ignore_order(): 
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed
    return ignore_order

def apply_to_img(func, ds):
    if func is not None: 
        ds = ds.map(lambda img, _: func(img), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_steps(df, batch_size): 
    return len(df) // batch_size


def build_dataset(paths, labels, decode_fn, img_transforms=None, batch_transforms=None, batch_size=4, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = apply_to_img(decode_fn, ds)
    ds = ds.cache().with_options(get_ignore_order())
    if is_training: 
        ds = ds.repeat().shuffle(512)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = apply_to_img(img_transforms, ds)
    ds = apply_to_img(batch_transforms, ds)
    return ds.prefetch(tf.data.AUTOTUNE) # Prefetch inputs before needed