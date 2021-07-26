from covid.tensorflow.augmentations.basic import get_basic_augmentations
from covid.tensorflow.augmentations.cutmix_mixup import get_cutmix_mixup
from covid.tensorflow.augmentations.gridmask import get_grid_mask
from covid.tensorflow.augmentations.augmix import get_augmix

def get_img_transforms(img_transforms, img_size, aug_params): 
    get_basic_augs, get_random_scale, get_random_rotate, get_random_cutout, get_random_blur, get_resize_fn = get_basic_augmentations()
    transform_string_to_transform = {
        'basic_augmentations': get_basic_augs(img_size), 
        'random_scale' : get_random_scale(img_size, aug_params['scale']), 
        'random_rotate' : get_random_rotate(img_size, aug_params['rot_prob']), 
        'random_cutout' : get_random_cutout(img_size, aug_params['cutout']), 
        'random_blur' : get_random_blur(aug_params['blur']), 
        'resize' : get_resize_fn(img_size), 
        'gridmask' : get_grid_mask(img_size, aug_params['gridmask']), 
        'augmix': get_augmix(img_size, aug_params['augmix']),
    }
    return [transform_string_to_transform[transform_str] for transform_str in img_transforms]


def get_batch_transforms(batch_transforms, img_size, aug_params, classes, batch_size): 
    cutmix, mixup = get_cutmix_mixup(img_size, classes, batch_size, cutmix_prob=aug_params['cutmix_prob'], mixup_prob=aug_params['mixup_prob'])
    transform_string_to_transform = {
        'cutmix': cutmix, 
        'mixup': mixup, 
    }
    return [transform_string_to_transform[transform_str] for transform_str in batch_transforms]
