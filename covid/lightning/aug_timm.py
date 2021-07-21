# Timm Transforms Imports
from timm.data.loader import *
from timm.data.transforms import *
from timm.data.transforms_factory import *
from timm.data.transforms import _pil_interp

# Primary Train Transforms
SCALE = (0.08, 1.0)
RATIO = (3./4., 4./3.)  
def primary_transforms(img_size, scale=SCALE, ratio=RATIO, vflip_prob=0.25): 
    return [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation='random'), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=vflip_prob),     
    ]

# Secondary Train Transforms
COLOR_JITTER = 0.4
def secondary_transforms(img_size, rand=None, augmix=None, color_jitter=COLOR_JITTER): 
    aa_params = {
        'translate_const': int(img_size * 0.45), 
        'img_mean': tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
    }
    secondary = []
    if rand is not None: 
        rand_aug = rand_augment_transform(rand, aa_params)
        secondary.append(rand_aug)
    if augmix is not None: 
        augmix_aug = augment_and_mix_transform(augmix, {**aa_params, 'translate_pct': 0.3})
        secondary.append(augmix_aug)
        
    color_jitter_aug = transforms.ColorJitter(COLOR_JITTER, COLOR_JITTER, COLOR_JITTER)
    secondary.append(color_jitter_aug)
    return secondary


# Final Train Transforms
RE_KWARGS = {
    'probability': 0.5, 'mode': 'const',
    'min_area': 0.02, 'max_area': 1/3, 'min_count': 1, 'max_count': None, 
    'device': 'cpu',
}
def final_transforms(re_prob=0.5, re_max_area=1/3, vis=False):
    RE_KWARGS['probability'] = re_prob
    RE_KWARGS['max_area'] = re_max_area
    final= [transforms.ToTensor()]
    if not vis: 
        final.append(
            transforms.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD)
            )
        )
    final.append(RandomErasing(**RE_KWARGS))
    return final
    