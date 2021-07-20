from pytorch_lightning.loggers import WandbLogger 

def get_wandb_logger(save_dir='wandb/', wandb_run='dummy_run', project='dummy', version=0): 
    save_dir = str(save_dir)
    wandb_logger = WandbLogger(
        save_dir = save_dir, 
        name = wandb_run, 
        offline = False, 
        project = project, 
        version = 0,  # To Resume previous run
    )
    return wandb_logger

"""

RUN_CHECKPOINT_DIR = './'
COMP_NAME = 'dummy'    

# TODO: Try overfit_batches: 0.01 to see if it works
# TODO: resume_from_checkpoint, callbacks, loggers  
RESUME_FROM_CHECKPOINT = 'last.pt'

# TRAINER CALLBACKS 
callbacks = []

# TRAINER LOGGERS
VERSION = 0
WANDB_RUN_NAME = f'backbone-name-version-v{VERSION}'

loggers = [
    get_wandb_logger(
        wandb_run = WANDB_RUN_NAME, 
        version = VERSION, 
        save_dir = RUN_CHECKPOINT_DIR / 'wandb', 
        project = COMP_NAME, 
    )
]    
trainer_kwargs = {
    # IMPORTANT
    'callbacks': callbacks, 
    'loggers': loggers, 
    'resume_from_checkpoint': str(RESUME_FROM_CHECKPOINT), 
    
    # AUTO TUNE
    'auto_scale_batch_size': 'binsearch',  # Find the largest batch size that will fit in memory
    'auto_lr_find': True, # Run learning rate finder
    # You have to call trainer.tune(model) after this
    
    # GRADIENTS 
    # use track_grad_norm to keep track of vanishing and exploding gradients
    # if you do find some gradients exploding, use gradient_clip_val to keep them from exploding to inf
    'gradient_clip_val': 0.0, # You need to fine tune it 
    'track_grad_norm': 2, # Track l2 norm
    'gradient_clip_algorithm': 'value', # Put 'norm' to clip by norm (better?)
    'accumulate_grad_batches': 1, # If data is large that it cannot fit in single batch, but you need large batch size
    
    # Limiting Time & Resources
    'max_epochs': 1000, 
    'min_epochs': 10, 
    'max_time': '00:10:00:00', # 10 hours
    
    # Debugging
    'num_sanity_val_steps': 2, # Run 2 batches of validation before starting training
    'reload_dataloaders_every_epoch': True, # ???
    'terminate_on_nan': True, # Terminate if anything is NaN 
    'weights_summary': 'full', # Full gives summary of all modules and submodules
    # after you know what this does, maybe change to 'top' to get summary for only top level modules
    'profiler': 'advanced', # ? les see what this does 

    # Free Performance
    'stochastic_weight_avg': True, 
    'benchmark': True, # Use this if input size is constant for the system(i'm assuming LitModel)
}

"""