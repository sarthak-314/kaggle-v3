
import multiprocessing
import torch
import os 


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

def create_loader(dataset, batch_size=32, is_training=False):
    using_tpu = 'TPU_NAME' in os.environ
    num_workers = 1 if using_tpu else optimal_num_of_loader_workers()
    
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


