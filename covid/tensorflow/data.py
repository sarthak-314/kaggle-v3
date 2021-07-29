import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
from time import time 
import pandas as pd

from covid.tensorflow.augmentations.basic import get_basic_augmentations
from covid.tensorflow.augmentations.cutmix_mixup import get_cutmix_mixup
from covid.tensorflow.augmentations.gridmask import get_grid_mask
from covid.tensorflow.augmentations.augmix import get_augmix

def _get_ignore_order(): 
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False # disable order, increase speed
    return ignore_order

def build_dataset(img_paths, labels, decode_fn, img_transforms=[], batch_transforms=[], batch_size=4, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().with_options(_get_ignore_order())
    if is_training: 
        ds = ds.repeat().shuffle(512)
    for img_transform in img_transforms: 
        ds = ds.map(img_transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    for batch_transform in batch_transforms: 
        ds = ds.map(batch_transform, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def _get_steps(df, batch_size): 
    steps = len(df) // batch_size
    return max(steps, 1)

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

def _get_img_transforms(img_transforms, img_size, aug_params): 
    get_basic_augs, get_random_scale, get_random_rotate, get_random_cutout, get_random_blur, get_resize_fn = get_basic_augmentations()
    transform_string_to_transform = {
        'basic_augmentations': get_basic_augs(img_size), 
        'random_scale' : get_random_scale(img_size, aug_params['scale']), 
        'random_rotate' : get_random_rotate(img_size, aug_params['rot_prob']), 
        'random_cutout' : get_random_cutout(img_size, aug_params['cutout']), 
        'random_blur' : get_random_blur(aug_params['blur']), 
        'resize' : get_resize_fn(img_size), 
        'gridmask' : get_grid_mask(img_size, aug_params['gridmask']), 
        'augmix': get_augmix(img_size, aug_params['augmix']),
    }
    return [transform_string_to_transform[transform_str] for transform_str in img_transforms]


def _get_batch_transforms(batch_transforms, img_size, aug_params, classes, batch_size): 
    cutmix, mixup = get_cutmix_mixup(img_size, classes, batch_size, cutmix_prob=aug_params['cutmix_prob'], mixup_prob=aug_params['mixup_prob'])
    transform_string_to_transform = {
        'cutmix': cutmix, 
        'mixup': mixup, 
    }
    return [transform_string_to_transform[transform_str] for transform_str in batch_transforms]


def get_train_ds_fn(img_size, img_ext, num_classes, aug_params, img_transforms, batch_transforms): 
    def get_train_ds(train, batch_size, debug_frac=1): 
        # Load train DataFrame
        if debug_frac != 1: 
            print(f'Taking {debug_frac*100}% of train ({int(len(train)*debug_frac)} samples), with batch size: {batch_size}')
            train = train.sample(frac=debug_frac)
        train = train.sample(frac=1)
        
        # Build train dataset
        start_time = time()
        train_ds = build_dataset(
            img_paths=train.img_path.values, 
            labels=pd.get_dummies(train.label).values, 
            decode_fn=get_decode_fn(img_ext, channels=3), 
            img_transforms=_get_img_transforms(img_transforms, img_size, aug_params), 
            batch_transforms=_get_batch_transforms(batch_transforms, img_size, aug_params, num_classes, batch_size), 
            batch_size=batch_size, 
            is_training=False, 
        )
        print(f'{time()-start_time} seconds to load train_ds')
        
        # train_steps 
        train_steps = _get_steps(train, batch_size)
        print(f'{(time()-start_time)*(train_steps/60)} minutes, {train_steps} steps for the first epoch ')
        return train_ds, train_steps
    return get_train_ds


def get_clean_ds_fn(img_size, img_ext, aug_params, img_transforms=['resize']):
    def get_clean_ds(df, batch_size, debug_frac=1): 
        # Load the dataframe
        if debug_frac != 1: 
            df = df.sample(frac=debug_frac)
        df = df.sample(frac=1)
        
        # Load the dataset 
        start_time = time()
        clean_ds = build_dataset(
            img_paths=df.img_path.values, 
            labels=pd.get_dummies(df.label).values, 
            decode_fn= get_decode_fn(img_ext, channels=3), 
            img_transforms=_get_img_transforms(img_transforms, img_size, aug_params), 
            batch_transforms=[], 
            batch_size=batch_size, 
            is_training=False, 
        )
        print(f'{time()-start_time} seconds to load clean ds')
        
        return clean_ds, _get_steps(df, batch_size)
    return get_clean_ds


def visualize_augmentations(should_visualize, train_ds, rows, cols): 
    if not should_visualize: 
        print('Skipping visualization')
        return 
    fig = plt.figure(figsize=(32, 12))
    num_imgs = rows * cols
    for i, (img, label) in tqdm(enumerate(train_ds.unbatch().take(num_imgs)), total=num_imgs): 
        if i < 5: print(f'label #{i}: ', label)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)