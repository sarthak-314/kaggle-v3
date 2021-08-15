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
TFHUB_URL = 'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-{backbone_size}-{weights}/feature-vector'


def get_effnetv2_from_tfhub(backbone_size):
    tfhub_url = TFHUB_URL.format(weights=WEIGHTS, backbone_size=backbone_size)
    return hub.KerasLayer(tfhub_url, trainable=True)

def get_effnetv2_from_github_repo(backbone_size): 
    kaggle_utils.startup.clone_repo('https://github.com/google/automl')
    os.chdir('./efficientnetv2/')
    import effnetv2_model
    model = f'efficientnetv2-{backbone_size}'
    return effnetv2_model.get_model()
    
    %cd ./efficientnetv2/
    import effnetv2_model
    def download_backbone(): 
        return effnetv2_model.get_model(
            f'efficientnetv2-{BACKBONE_SIZE}', 
            include_top=INCLUDE_TOP, 
            weights=None,  
            training=True, 
            with_endpoints=WITH_ENDPOINTS, 
        )
    
    
    
def download_effnetv2(backbone_size, backbone_source, with_endpoints=True, include_top=False):
    if backbone_source == 'TFHub': 
        print(f'Downloading effnetv2-{backbone_size} from TFHub')
        return get_effnetv2_from_tfhub(backbone_size)
        
    elif backbone_source == 'Repo':
        print(f'Cloning and downloading effnetv2-{backbone_size} from automl repo')
        
    





assert BACKBONE_NAME == 'efficientnetv2'
WEIGHTS_LOAD_PATH = 

# Backbone Config
BACKBONE_SOURCE = 'Repo' 
WEIGHTS = 'imagenet21k-ft1k'

# Only Repo Config
INCLUDE_TOP = False
WITH_ENDPOINTS = True

# Backbone Info
model_size_to_img_size = {'s': 384, 'm': 480, 'l': 480, 'xl': 480}
print('Original Model Img Size: ', colored(model_size_to_img_size[BACKBONE_SIZE], 'blue'))

# Load Backbone from Source
if BACKBONE_SOURCE == 'TFHub': 
    
    def download_backbone(): 
        return hub.KerasLayer(tfhub_url, trainable=True)

elif BACKBONE_SOURCE == 'Repo': 


# Backbone Utils
def save_only_backbone_weights(filepath): 
    pass

# Backbone Setup 
with STRATEGY.scope():
    backbone = download_backbone()
    backbone.load_weights(WEIGHTS_LOAD_PATH)

def build_backbone(img_size, with_endpoints=WITH_ENDPOINTS): 
    with STRATEGY.scope(): 
        backbone(
            tf.keras.Input((img_size, img_size, 3)), 
            with_endpoints=with_endpoints, 
            training=True, 
        )
        print(f'Building backbone with size {BACKBONE_SIZE} & Image Size: {img_size}')
        #backbone.summary()