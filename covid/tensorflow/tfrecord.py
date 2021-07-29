from tqdm.auto import tqdm
import tensorflow as tf

def compress_img(img, label):
    img = tf.cast(img, tf.uint8)
    img = tf.io.encode_jpeg(img, optimize_size=True)
    return img, label

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def get_tfrec_dataset(img_paths, labels, shard_size, decode_fn, resize_fn):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(compress_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(shard_size, drop_remainder=False)
    return ds

def to_tfrecord(tfrec_filewriter, img_bytes, label):
    feature = {
        "img": _bytestring_feature([img_bytes]), 
        "label": _int_feature([label]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def get_decode_fn(img_extension, channels):
    def decode_fn(path, label): 
        file_bytes = tf.io.read_file(path)
        if img_extension == 'png':
            img = tf.image.decode_png(file_bytes, channels=channels)
        elif img_extension in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=channels)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    return decode_fn

def build_tfrecords(df, img_size, shard_size, tfrec_dir, ext='png'): 
    decode_fn = get_decode_fn(ext, 3) 
    def resize_fn(img, label): 
        img = tf.image.resize(img, size=[img_size, img_size])
        return img, label
    ds = get_tfrec_dataset(df.img_path.values, df.label.values, shard_size, decode_fn, resize_fn)
    for shard, (img, label) in tqdm(enumerate(ds), total=len(df)//shard_size):
        filename = str(tfrec_dir / f'{shard}-{shard_size}.tfrec')
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = to_tfrecord(out_file, img.numpy()[i], label.numpy()[i])
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))
            
def get_tfrec_builder(img_size, shard_size, tfrec_dir): 
    def tfrec_builder(df, ext): 
        return build_tfrecords(df, img_size, shard_size, tfrec_dir, ext=ext)
    return tfrec_builder