import tensorflow as tf
def _bytes_feature(value): 
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(img, label): 
    feature = {
        'img': _bytes_feature(img), 
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(img, label): 
    tf_string = tf.py_function(
        serialize_example, 
        (img, label), 
        tf.string
    )
    return tf.reshape(tf_string, ())

def get_serialized_feature_dataset(paths, labels, decode_fn): 
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
