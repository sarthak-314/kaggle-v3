
# Light Augmentation Params
AUG_PARAMS = {
    'scale': { 'zoom_in': 0.75, 'zoom_out': 0.9, 'prob': 0.5 }, 
    'rot_prob': 0.75, 'blur': { 'ksize': 2, 'prob': 0.05 }, 
    'gridmask': { 'd1': 20,  'd2': 120,  'rotate': 90,  'ratio': 0.5,  'prob': 0.25 },
    'cutout': { 'sl': 0.01, 'sh': 0.1,  'rl': 0.5, 'prob': 0.2 }, 
    'cutmix_prob': 0.25, 'mixup_prob': 0.25, 
    # 'augmix': { 'severity': 1, 'width': 2, 'prob': 0.5 },  
}

IMG_TRANSFORMS = [ 
    'basic_augmentations', 'random_scale', 'resize', 
    'random_cutout', 'random_rotate', 'gridmask'
]
BATCH_TRANSFORMS = ['cutmix', 'mixup']




