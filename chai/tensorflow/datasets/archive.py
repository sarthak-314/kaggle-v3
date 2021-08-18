################################ TASK 1 PNG #######################
"""
- 3m 16s  to visualize images

IDEAS TO TRY
- Scale the data 3x by including all the axis
"""
# Dataset Config
MAX_BLACK_PER = 0.25
IMGS_PER_EPOCH = 2000

# Build Dataframes
GCS_PATH = 'gs://kds-49fd7c0dfe8cbb3ac5d1a4ad43a056b1b32d4e642283a0013f180376'
to_train, to_valid = read_dataframes(FOLD)
# to_train, to_valid = to_train[to_train.black_per < MAX_BLACK_PER], to_valid[to_valid.black_per < MAX_BLACK_PER]
to_train.img_path = to_train.img_path.replace('/kaggle/input/tumor-data-task-1-png', GCS_PATH)
to_valid.img_path = to_valid.img_path.replace('/kaggle/input/tumor-data-task-1-png', GCS_PATH)

# Utility Functions for building datasets
decode_fn = get_decode_fn('png', channels=3)

def get_one_hot_label(img_path, int_label):
    one_hot_length = 2
    one_hot_label = tf.one_hot(int_label, one_hot_length)
    return img_path, tf.cast(one_hot_label, tf.float32)

def resize_fn(img, label): 
    img = tf.image.resize(img, size=[IMG_SIZE, IMG_SIZE])
    return img, label


def get_train_ds(img_paths, labels, batch_size): 
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(get_one_hot_label, **AUTO)
    ds = ds.map(decode_fn, **AUTO).map(resize_fn, **AUTO)

    # Image Augmentations
    for img_transform in get_img_transforms(IMG_TRANSFORMS, IMG_SIZE, AUG_PARAMS): 
        ds = ds.map(img_transform, **AUTO).map(lambda img, label: (tf.squeeze(img), label), **AUTO)
    ds = ds.map(lambda img, label: (tf.reshape(img, [IMG_SIZE, IMG_SIZE, CHANNELS]), label), **AUTO)
    ds = ds.repeat().shuffle(4096).batch(batch_size)
    for batch_transform in get_batch_transforms(BATCH_TRANSFORMS, IMG_SIZE, AUG_PARAMS, 2, batch_size): 
        ds = ds.map(batch_transform, **AUTO)

    # Dummy Masks
    if ADD_DUMMY_MASK: 
        dummy_mask_ds = tf.data.Dataset.from_tensors(tf.zeros((batch_size, IMG_SIZE, IMG_SIZE))).repeat()
        img_ds, label_ds = ds.map(lambda img, label: img, **AUTO), ds.map(lambda img, label: label, **AUTO)
        output_ds = tf.data.Dataset.zip((label_ds, dummy_mask_ds))
        ds = tf.data.Dataset.zip((img_ds, output_ds))
    
    return ds.prefetch(tf.data.AUTOTUNE)

def get_valid_ds(img_paths, labels, batch_size): 
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(get_one_hot_label, **AUTO)
    ds = ds.map(decode_fn, **AUTO).map(resize_fn, **AUTO)
    ds = ds.repeat().shuffle(512).batch(batch_size)
    if ADD_DUMMY_MASK: 
        dummy_mask_ds = tf.data.Dataset.from_tensors(tf.zeros((batch_size, IMG_SIZE, IMG_SIZE))).repeat()
        img_ds, label_ds = ds.map(lambda img, label: img, **AUTO), ds.map(lambda img, label: label, **AUTO)
        output_ds = tf.data.Dataset.zip((label_ds, dummy_mask_ds))
        ds = tf.data.Dataset.zip((img_ds, output_ds))
    return ds.prefetch(tf.data.AUTOTUNE)

# ------------------------- AUGMENTATIONS & DEBUGGING --------------------------
# ------------------------------------------------------------------------------
VISUALIZE = False

# Very Light Augmentation Params
AUG_PARAMS = {
    'scale': { 'zoom_in': 0.75, 'zoom_out': 0.95, 'prob': 0.5 }, 
    'rot_prob': 0.75, 'blur': { 'ksize': 2, 'prob': 0.05 }, 
    'gridmask': { 'd1': 20,  'd2': 120,  'rotate': 90,  'ratio': 0.5,  'prob': 0.1 },
    'cutout': { 'sl': 0.01, 'sh': 0.1,  'rl': 0.5, 'prob': 0.1 }, 
    'cutmix_prob': 0.05, 'mixup_prob': 0.1, 
    'augmix': { 'severity': 1, 'width': 2, 'prob': 0 },  
}

IMG_TRANSFORMS = [ 
    # 'basic_augmentations', 
    'random_scale', 'resize', 'random_rotate'
    # 'random_cutout', , 'gridmask'
]
# BATCH_TRANSFORMS = ['cutmix', 'mixup']
BATCH_TRANSFORMS = ['mixup']

# Visualize Augmentations
ROWS, COLS = 4, 8
train_ds = get_datasets(COLS)[0]
print(train_ds.take(1))
visualize_augmentations(VISUALIZE, train_ds, rows=ROWS, cols=COLS)









################################ 3D NII DATASET ##############################
