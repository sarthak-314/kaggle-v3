from tqdm.auto import tqdm
import tensorflow as tf

def compress_img(img):
    img = tf.cast(img, tf.uint8)
    img = tf.io.encode_jpeg(img, optimize_size=True)
    return img

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def get_tfrec_dataset(img_paths, labels, shard_size, decode_fn, resize_fn):
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels) 
    img_ds = path_ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    img_ds = img_ds.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    img_ds = img_ds.map(compress_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    ds = ds.batch(shard_size, drop_remainder=False)
    return ds

def to_tfrecord(tfrec_filewriter, img_bytes, label):
    feature = {
        "img": _bytestring_feature([img_bytes]), 
        "label": _int_feature([label]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def build_tfrecords(df, img_size, shard_size, tfrec_dir): 
    png_df = df[df.img_ext == 'png']
    jpg_df = df[(df.img_ext == 'jpeg') | (df.img_ext == 'jpg')]
    png_decode_fn = lambda path: tf.image.decode_png(path, channels=3)
    jpg_decode_fn = lambda path: tf.image.decode_jpeg(path, channels=3)
    resize_fn = lambda img: tf.image.resize(img, size=[img_size, img_size])
    png_ds = get_tfrec_dataset(png_df.img_path.values, png_df.label.values, png_decode_fn, resize_fn)
    jpg_ds = get_tfrec_dataset(jpg_df.img_path.values, jpg_df.label.values, jpg_decode_fn, resize_fn)
    for ds in png_ds, jpg_ds: 
        for shard, (img, label) in tqdm(enumerate(ds), total=len(df)//shard_size):
            filename = str(tfrec_dir / f'{shard}-{shard_size}.tfrec')
            with tf.io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    example = to_tfrecord(out_file, img.numpy()[i], label.numpy()[i])
                    out_file.write(example.SerializeToString())
                print("Wrote file {} containing {} records".format(filename, shard_size))