import matplotlib.pyplot as plt
from tqdm.auto import tqdm
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
    steps = len(df) // batch_size
    return max(steps, 1)

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