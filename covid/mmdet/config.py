def add_wandb_hook(use_wandb, cfg, project, run_name):
    if not use_wandb:
        print('Not using WandB') 
    else: 
        cfg.log_config.hooks.append(dict(type='WandbLoggerHook', init_kwargs=dict(project=project, name=run_name)))
    return cfg
