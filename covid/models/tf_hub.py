from covid.startup import *

KAGGLE_DATASET = 'siim-covid19-resized-to-1024px-jpg'




def get_img_path(img_id): 
    return f'{GSC_PATH}/train/{img_id}.jpg'

def get_datasets(batch_size):
    train_img_paths = train.img_id.apply(get_img_path).values
    train_labels = train[LABEL_COLS].values
    train_ds = build_comp_dataset(train_img_paths, train_labels, img_size=IMG_SIZE, augment_fn=comp_augment_fn, batch_size=batch_size)

    valid_img_paths = valid.img_id.apply(get_img_path).values
    valid_labels = valid[LABEL_COLS].values
    valid_ds = build_comp_dataset(valid_img_paths, valid_labels, img_size=IMG_SIZE, augment_fn=None, batch_size=batch_size)

    train_steps, valid_steps = len(train) // batch_size, len(valid) // batch_size
    return train_ds, valid_ds, train_steps, valid_steps


# %%
def build_model(backbone_trainable=True): 
    backbone_url = f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/{BACKBONE_NAME}/feature-vector'
    backbone = hub.KerasLayer(backbone_url, trainable=backbone_trainable, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    print('backbone downloaded')
    backbone.trainable = backbone_trainable
    model = tf.keras.Sequential([
        backbone, 
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

def compile_model(model, lr): 
    model.compile(
        optimizer=get_ranger(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'], 
        steps_per_execution=32
    )

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(
        filepath=path, 
        options=get_save_locally(), 
    )
    if 'freeze' in str(path): 
        name = 'freeze'
    else: 
        name = 'trained'
    weights_path = str(path/f'{name}.h5')
    model.save_weights(
        filepath=weights_path, 
        options=get_save_locally(), 
    )
    wandb.save(weights_path)

def get_callbacks(use_wandb=False):
    callbacks_list = [
        get_model_checkpoint(CHECKPOINT_DIR), 
        get_early_stopping(), 
        get_reduce_lr_on_plateau(), 
        time_stopping(MAX_TRAIN_HOURS), 
        tqdm_bar(), 
        terminate_on_nan(), 
    ]
#     if use_wandb:
#         from wandb.keras import WandbCallback
#         callbacks_list.append(WandbCallback())
    return callbacks_list


# %%
BATCH_SIZE = 256
USE_WANDB = True

EPOCHS_FREEZE = 50
LR_FREEZE = 1e-3

EPOCHS_TRAIN = 100
LR_TRAIN = 1e-4


train_ds, valid_ds, train_steps, valid_steps = get_datasets(BATCH_SIZE)
with STRATEGY.scope(): 
    # General Training Setup
    callbacks = get_callbacks(USE_WANDB)
    fit_kwargs = {
        'x': train_ds, 
        'verbose': 1, 
        'callbacks': callbacks, 
        'validation_data': valid_ds, 
        'steps_per_epoch': train_steps, 
        'validation_steps': valid_steps, 
    }

    # Freeze Training 
    model = build_model(backbone_trainable=False)
    model.summary()
    compile_model(model, LR_FREEZE)
    print('fitting model')
    history = model.fit(
        **fit_kwargs, 
        epochs=EPOCHS_FREEZE, 
    )
    freezed_save_path = CHECKPOINT_DIR / 'freezed' 
    save_model(model, freezed_save_path)
    
    
    # Full Training
#     model = build_model(backbone_trainable=True)
    model.trainable = True
    compile_model(model, LR_TRAIN)
    print('fitting model')
    history = model.fit(
        **fit_kwargs, 
        epochs=EPOCHS_TRAIN, 
    )
    save_model(model, CHECKPOINT_DIR/'trained')


# %%



# %%



