import tensorflow as tf
try: 
    import tensorflow_hub as hub
except: 
    print('Tensorflow Hub not availible')
try: 
    import tensorflow_addons as tfa 
except: 
    print('Tensorflow addons not availible')

from termcolor import colored 


# Full Module Imports 
from chai.tflow.callbacks import * 
AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }

##### Startup Functions #####
def tf_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def mixed_precision(): 
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def jit(): 
    tf.config.optimizer.set_jit(True)


##### Important Functions #####
def save_weights(model, filepath):
    filepath = str(filepath)
    print('Saving model weights at', colored(filepath, 'blue'))
    model.save_weights(filepath=filepath, options=get_save_locally())

def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)
    
def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')


##### Factory Functions #####
def tf_lr_scheduler_factory(lr_scheduler_kwargs): 
    if isinstance(lr_scheduler_kwargs, float): 
        print(colored('Using constant learning rate', 'yellow'))
        return lr_scheduler_kwargs
    lr_scheduler = tfa.optimizers.ExponentialCyclicalLearningRate(
        initial_learning_rate=lr_scheduler_kwargs['init_lr'], 
        maximal_learning_rate=lr_scheduler_kwargs['max_lr'], 
        step_size=lr_scheduler_kwargs['step_size'], 
        gamma=lr_scheduler_kwargs['gamma'], 
    )
    return lr_scheduler

def tf_optimizer_factory(optimizer_kwargs, lr_scheduler): 
    optimizer_name = optimizer_kwargs['name']
    if optimizer_name == 'AdamW': 
        optimizer = tfa.optimizers.AdamW(
            weight_decay=optimizer_kwargs['weight_decay'],
            learning_rate=lr_scheduler,  
            amsgrad=False, 
            clipnorm=optimizer_kwargs['max_grad_norm'], 
        )
    elif optimizer_name == 'Adagrad': 
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=lr_scheduler, 
        )
        print('Skipping weight decay for Adagrad')
    if optimizer_kwargs['use_lookahead']: 
        print(colored('Using Lookahead', 'red'))
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if optimizer_kwargs['use_swa']: 
        print(colored('Using SWA', 'red'))
        optimizer = tfa.optimizers.SWA(optimizer)
    return optimizer



