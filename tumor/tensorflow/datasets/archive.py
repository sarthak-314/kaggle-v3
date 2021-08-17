################################ 3D NII DATASET ##############################
"""
- 3m 30s + 45s to Visualize Images (TPU)
BUG: Divide by 5885. for normalization

IDEAS 
- Taking random slice from a 3D array 
- Filtering images with low information by taking black pixel count
- Taking 3 consecutive slices to make 3 channels

IDEAS TO TRY
- Try 16 channel input (easy to do if backbone supports it)
- In a TFRecord, make it 'FLAIR', 'T1w', 'T1wce', 'T2', 'seg' and make multi channel image with overlapping flair, t1w, etc. 
- Better than taking 3 consecutive slices
"""
sync()
import tumor.tensorflow.datasets.task_one as task_one

# Dataset Builder Config
IMGS_PER_EPOCH = 4000
SLICES_PER_NII = 64
MAX_BLACK_RATIO = 0.25
MIN_DEPTH = 16

# TFRecords Source (tumor-task1-foldx-tfrecs)
TASK1_GCS_PATHS = [ 
    'gs://kds-d893ba40b104bc98fafde3615607654a0da18a0738e8ce2fbe4993c1',
    'gs://kds-36e2bc1ed9cc2721bd42544b7be16b2f1c59ef6cbdb0559f25526084',
    'gs://kds-e7dc052159590b3723d12efb3215e9cc4458e49e034661c6c55b4767',
    'gs://kds-130e0e32262624592b6f504fdd47eb2be2f06855a047aa9bf9ddf493',
]
print(TASK1_GCS_PATHS)

def get_random_slice(img_3d, seg_3d, depth): 
    # Cut Random Slice
    MINVAL, MAXVAL = 50, 200
    # if tf.random.uniform([], 0, 1.0, dtype=tf.float32) < 0.25:
    #     if tf.random.uniform([], 0, 1.0, dtype=tf.float32) < 0.5:
    #         random_slice = tf.random.uniform(shape=[], minval=MINVAL, maxval=MAXVAL, dtype=tf.int64)
    #         img, seg = img[random_slice:random_slice+3, :, :], seg[random_slice:random_slice+3, :, :]
    #         img, seg = tf.transpose(img, [1, 2, 0]), tf.transpose(seg, [1, 2, 0])
    #     else: 
    #         random_slice = tf.random.uniform(shape=[], minval=MINVAL, maxval=MAXVAL, dtype=tf.int64)
    #         img, seg = img[:, random_slice:random_slice+3, :], seg[:, random_slice:random_slice+3, :]
    #         img, seg = tf.transpose(img, [0, 2, 1]), tf.transpose(seg, [0, 2, 1])
    # else: 
    minval = depth//10
    random_slice = tf.random.uniform(shape=[], minval=minval, maxval=depth-3, dtype=tf.int64)
    img = img_3d[:, :, random_slice:random_slice+3]
    seg = seg_3d[:, :, random_slice:random_slice+3]
    seg = seg[:, :, 1]
    return img, seg

def crop_center(img, seg):
    seg = tf.expand_dims(seg, axis=-1)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    seg = tf.image.resize_with_crop_or_pad(seg, IMG_SIZE, IMG_SIZE)
    img = tf.reshape(img, (IMG_SIZE, IMG_SIZE, 3))
    seg = tf.reshape(seg, (IMG_SIZE, IMG_SIZE))
    return img, seg

def get_random_slices(img_3d, seg_3d, depth): 
    imgs, segs = [], []
    for _ in range(SLICES_PER_NII): 
        img, seg = get_random_slice(img_3d, seg_3d, depth)
        img, seg = crop_center(img, seg)
        imgs.append(img); segs.append(seg)
    return tf.stack(imgs), tf.stack(segs)

def filter_empty_imgs(img, seg): 
    take_img = (tf.math.count_nonzero(img[:, :, 0]) / IMG_SIZE**2) > MAX_BLACK_RATIO
    # print('take_img: ', take_img)
    return take_img

def normalize(img, seg): 
    'Normalize and make boolean masks'
    img = img / tf.reduce_max(img) # Self normalize each image
    seg = seg / tf.reduce_max(seg)
    return img, seg


# Load the TFRecords
task1_train_tfrecs, task1_valid_tfrecs = task_one.load_tfrecords(TASK1_GCS_PATHS, FOLD)
tfrecs = task1_train_tfrecs # Skip patients which are in validation for task 2

tfrec_datasets = task_one.build_datasets_from_tfrecs(tfrecs)
img_seg_depth_ds = tf.data.Dataset.zip((tfrec_datasets['img'], tfrec_datasets['segmentation'], tfrec_datasets['depth']))
img_seg_ds = img_seg_depth_ds.map(get_random_slices, **AUTO).unbatch()
img_seg_ds = img_seg_ds.map(normalize, **AUTO)
img_seg_ds = img_seg_ds.filter(filter_empty_imgs)
img_ds = img_seg_ds.map(lambda img, seg: img, **AUTO)
seg_ds = img_seg_ds.map(lambda img, seg: seg, **AUTO)

label_ds = tfrec_datasets['label_int'].map(lambda int_label: tf.one_hot(int_label, 2), **AUTO)

inputs = (img_ds)
input_ds = tf.data.Dataset.zip(inputs)
outputs = (label_ds, seg_ds)
output_ds = tf.data.Dataset.zip(outputs)

ds = tf.data.Dataset.zip((input_ds, output_ds))

def get_task1_dataset(batch_size): 
    task1_train_ds = ds.repeat().shuffle(2048).batch(batch_size)
    return task1_train_ds.prefetch(tf.data.AUTOTUNE)

# Visualize Images
rows, cols = 4, 8
task1_train_ds = get_task1_dataset(cols)
fig = plt.figure(figsize=(32, 12))
num_imgs = rows * cols
xs = []
for i, x in tqdm(enumerate(task1_train_ds.unbatch().take(num_imgs)), total=num_imgs): 
    _ = fig.add_subplot(rows, cols, i+1)
    _ = plt.imshow(tf.expand_dims(x[1][1], axis=-1) + x[0])
    xs.append(x)
