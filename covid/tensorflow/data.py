import tensorflow as tf

def get_ignore_order(): 
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed
    return ignore_order

def apply_transforms(ds, transforms): 
    AUTO = tf.data.experimental.AUTOTUNE
    if transforms is not None: 
        ds = ds.map(transforms, num_parallel_calls=AUTO)
    return ds


def get_steps(df, batch_size): 
    return len(df) // batch_size