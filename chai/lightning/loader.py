
import torch
import os 

def create_loader(dataset, batch_size=32, is_training=False):
    using_tpu = 'TPU_NAME' in os.environ
    num_workers = 1 if using_tpu else 4
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_training, 
        drop_last=is_training, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=False,
    )
    return data_loader