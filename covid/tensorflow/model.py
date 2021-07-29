import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow as tf
import wandb
import os 

def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def _get_ranger(ranger_kwargs): 
    radam = tfa.optimizers.RectifiedAdam( 
        lr=ranger_kwargs['lr'],
        total_steps=ranger_kwargs['total_steps'], 
        warmup_proportion=ranger_kwargs['warmup_proportion'], 
        amsgrad=ranger_kwargs['amsgrad'],
        min_lr=0,
        weight_decay=1e-4, # default is 0
        #clipnorm = 10, # Not present by befault
        name='Ranger',
    )
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    return ranger

def build_model(tfhub_url, z_dim, num_dense, dropout=0.5): 
    model = tf.keras.Sequential([
        hub.KerasLayer(tfhub_url, trainable=True), 
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(
            z_dim, 
            activation=tfa.activations.mish, 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(num_dense, kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    ])
    return model

def compile_model(model, strategy, loss, metrics=['accuracy'], ranger_kwargs=None): 
    with strategy.scope(): 
        model.compile(
            loss=loss,
            metrics=metrics, 
            optimizer=_get_ranger(ranger_kwargs), 
            steps_per_execution=32,
        )
    print('Model Compiled')


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(filepath=path, options=get_save_locally())
    model.save_weights(filepath=str(path/f'weights.h5'), options=get_save_locally())
    try: 
        wandb.save(str(path/'weights.h5'))
    except: 
        print('Skipping wandb save')

