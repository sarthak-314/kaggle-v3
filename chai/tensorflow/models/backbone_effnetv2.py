"""
Efficient Net V2
----------------
- Trains faster and more accurate than V1 and NfNet
- Trained with Progressive Learning and Progressive Regularization
- Small Img Size + Low Regularization -> Larger Img Size + High Regularization

| Model Size  | Image Size |
| ----------- | ---------- |
|     s       |    384     |
|     m       |    480     |
|     l       |    480     |
|     xl      |    480     |

Availible Sizes: s, m, l, xl
Availible Weights: imagenet21k, imagenet21k-ft1k
"""
import tensorflow_hub as hub
import tensorflow as tf 
import os

import kaggle_utils.startup

WEIGHTS = 'imagenet21k-ft1k' 
INCLUDE_TOP = False # Skip the final dense layer
WITH_ENDPOINTS = True # Return feature vector when cloned from Repo

TFHUB_URL = 'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-{backbone_size}-{weights}/feature-vector'

def get_effnetv2_from_tfhub(backbone_size):
    tfhub_url = TFHUB_URL.format(weights=WEIGHTS, backbone_size=backbone_size)
    return hub.KerasLayer(tfhub_url, trainable=True)

def get_effnetv2_from_github_repo(backbone_size): 
    kaggle_utils.startup.clone_repo('https://github.com/google/automl')
    os.chdir('./efficientnetv2/')
    import effnetv2_model
    model = f'efficientnetv2-{backbone_size}'
    return effnetv2_model.get_model(
        f'efficientnetv2-{backbone_size}', 
        include_top=INCLUDE_TOP, 
        weights=None,  
        training=True, 
        with_endpoints=WITH_ENDPOINTS, 
    )

def get_effnetv2_from_kaggle_repo(): 
    pass


def download_effnetv2(backbone_size, backbone_source):
    if backbone_source == 'TFHub': 
        print(f'Downloading effnetv2-{backbone_size} from TFHub')
        return get_effnetv2_from_tfhub(backbone_size)
    elif backbone_source == 'Repo':
        print(f'Cloning and downloading effnetv2-{backbone_size} from automl repo')
        return get_effnetv2_from_github_repo(backbone_size)