import tensorflow as tf

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE } 

def get_ignore_order(): 
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed
    return ignore_order