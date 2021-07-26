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


