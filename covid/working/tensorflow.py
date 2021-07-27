from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
DIR, NAME = user_secrets.get_secret('DIR'), user_secrets.get_secret('NAME')
import subprocess
subprocess.run(['git', 'clone', NAME, DIR])


FOLD_NUMBER_TO_TAKE = 0
BACKBONE_TO_TRAIN, IMAGE_SIZE_TRAIN = 'efficientnetv2-xl-21k-ft1k', 512
HUBHUBN_URL = f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/{BACKBONE_TO_TRAIN}/feature-vector'
sync(); train, valid, test = read_dataframes(FOLD_NUMBER_TO_TAKE)
THE_SAVE_PATH_DIR = WORKING_DIR / 'Checkpoints'

# Tensorflow Setup
STRATEGY = auto_select_accelerator()
# enable_mixed_precision() ## BUG: Loading from TFHub
# WEIGHTS_PATH = KAGGLE_INPUT_DIR/'tensorflow-models'/'effnetv2_cxr100.h5'
def build_model(dropout=0.5, num_dense=4): 
    model = tf.keras.Sequential([
        hub.KerasLayer(HUBHUBN_URL, trainable=True), 
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(num_dense, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
    return model


def load_model(weights_path, final_layers, dropout): 
    with STRATEGY.scope(): 
        model = build_model(dropout, final_layers)
        model.build((None, IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN, 3)); model.summary()
        model.load_weights(weights_path) 
        model.layers[0].trainable = True
        for layer in model.layers: layer.trainable = True
        # model.pop(); model.add(tf.keras.layers.Dense(4))
    return model
def build_dataset(paths, labels, decode_fn, img_transforms=[], batch_transforms=[], batch_size=4, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().with_options(get_ignore_order())
    if is_training: ds = ds.repeat().shuffle(512)
    
    for img_transform in img_transforms: 
        ds = ds.map(img_transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    for batch_transform in batch_transforms: 
        ds = ds.map(batch_transform, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def get_train_ds(train, batch_size, debug_frac=1):
    start_time = time()
    if debug_frac != 1: 
        print(f'Taking {debug_frac*100}% of train ({int(len(train)*debug_frac)} samples), with batch size: {batch_size}')
        train = train.sample(frac=debug_frac)
    train_img_transforms = get_img_transforms(IMG_TRANSFORMS, IMAGE_SIZE_TRAIN, AUG_PARAMS)
    batch_transforms_fns = get_batch_transforms(BATCH_TRANSFORMS, IMAGE_SIZE_TRAIN, AUG_PARAMS, NUM_CLASSES, batch_size)
    img_paths, labels = train.img_path.values, pd.get_dummies(train.label).values
    train_ds = build_dataset(img_paths, labels, decode_fn, train_img_transforms, batch_transforms_fns, batch_size, True)
    print(f'{time()-start_time} seconds to load train_ds')
    train_steps = get_steps(train, batch_size)
    print(f'{(time()-start_time)*(train_steps/60)} minutes, {train_steps} steps for the first epoch ')
    return train_ds, train_steps

def get_clean_ds(df, batch_size, debug_frac=1): 
    start_time = time()
    if debug_frac != 1: df = df.sample(frac=debug_frac)
    df_img_transforms = get_img_transforms(['resize'], IMAGE_SIZE_TRAIN, AUG_PARAMS)
    batch_transforms = []
    img_paths, labels = df.img_path.values, pd.get_dummies(df.label).values
    dataset = build_dataset(img_paths, labels, decode_fn, df_img_transforms, batch_transforms, batch_size, False)
    return dataset, get_steps(df, batch_size)

def get_datasets(batch_size, debug_frac): 
    train_ds, train_steps = get_train_ds(train, batch_size, debug_frac)
    valid_ds, valid_steps = get_clean_ds(valid, batch_size, debug_frac)
    return train_ds, train_steps, valid_ds, valid_steps

KAGGLE_DATASET, NUM_CLASSES, IMG_EXT = 'siim-covid19-resized-to-256px-png', 4, 'png'
DATASET_DIR, decode_fn = KAGGLE_INPUT_DIR/KAGGLE_DATASET, get_decode_fn(IMG_EXT, 3)

# Augmentations Hyperparameters
IMAGE_SIZE_TRAIN = 128
SAMPLING_STRATEGY = None # Oversampling, Undersampling
OVERSAMPLE_TIMES = {0: 1, 1: 1, 2: 1, 3: 1} 
CLASS_WEIGHTS = { 0: 1, 1: 0.75, 2: 1.25, 3: 2 }
AUG_PARAMS = {
    'scale': { 'zoom_in': 0.5, 'zoom_out': 0.9, 'prob': 0.25 }, 
    'rot_prob': 0.5, 'blur': { 'ksize': 5, 'prob': 0.05 }, 
    'gridmask': { 'd1': 5,  'd2': 20,  'rotate': 90,  'ratio': 0.5,  'prob': 0.05 },
    'cutout': { 'sl': 0.01, 'sh': 0.05,  'rl': 0.5, 'prob': 0.1 }, 
    'cutmix_prob': 0.25, 'mixup_prob': 0.1, 
    'augmix': { 'severity': 1, 'width': 2, 'prob': 0 },  
}
IMG_TRANSFORMS = [ 'basic_augmentations', 'random_scale', 'resize', 'random_rotate', 'gridmask', 'random_cutout', 'augmix' ]
BATCH_TRANSFORMS = ['cutmix', 'mixup']

# Build Training & Validation Dataframes
train, valid, test = read_dataframes(FOLD_NUMBER_TO_TAKE)
train, valid = add_kaggle_and_gcs_path(train, valid, KAGGLE_DATASET)
train['img_path'], valid['img_path'] = train.gcs_path, valid.gcs_path
# Oversample or Undersample
# oversample(train, OVERSAMPLE_TIMES)

# train = oversample(train, OVERSAMPLE_TIMES)
VISUALIZE, ROWS, COLS = True, 4, 8
visualize_augmentations(VISUALIZE, get_train_ds(train, COLS)[0], rows=ROWS, cols=COLS)
# WEIGHTS_PATH = '/kaggle/working/Checkpoints/freezed_500/weights.h5'
DROPOUT = 0.9
with STRATEGY.scope(): 
    model = build_model(DROPOUT, 4)
    model.build((None, IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN, 3)); model.summary()
    model.load_weights(WEIGHTS_PATH) 
    # model.pop(); model.add(tf.keras.layers.Dense(4))
    model.layers[0].trainable = True
    for layer in model.layers: layer.trainable = True

def complie_the_model(model): 
    with STRATEGY.scope():
        model.compile(
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)], 
            optimizer=get_ranger(1e-2), steps_per_execution=32,
        )
    print('Model Compiled')
    
# (5m + 5m + 0.5s/epoch) Freeze Training (No Augmentations; Imbalanced Class Weights)
EPOCHS, BATCH_SIZE, DEUG_FRAC = 500, 4096, 1
clean_train_ds, train_steps = get_clean_ds(train, BATCH_SIZE, DEUG_FRAC)

with STRATEGY.scope(): model.layers[0].trainable = False
model.summary(); complie_the_model(model)
model.fit(
    clean_train_ds, steps_per_epoch=train_steps, epochs=EPOCHS, 
    class_weight={0: 1, 1: 0.25, 2: 2, 3: 3}, callbacks=[get_reduce_lr_on_plateau(100)], 
)
wandb.init('model-saver-effnetv2')
save_model(model, THE_SAVE_PATH_DIR/'freezed_500')
wandb.save('/kaggle/working/Checkpoints/freezed_500/weights.h5')