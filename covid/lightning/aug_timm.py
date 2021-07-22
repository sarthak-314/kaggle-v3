# Timm Transforms Imports
from timm.data.loader import *
from timm.data.transforms import *
from timm.data.transforms_factory import *
from timm.data.transforms import _pil_interp
import matplotlib.pyplot as plt

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
    'min_area': 0.02, 'max_area': 1/3, 'min_count': 1, 'max_count': 4, 
    'device': 'cpu',
}
def final_transforms(re_prob=0.5, max_count=4, max_area=1/3, vis=False):
    RE_KWARGS['probability'] = re_prob
    RE_KWARGS['max_area'] = max_area
    RE_KWARGS['max_count'] = max_count
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
    

def get_eval_transforms(img_size, crop_pct): 
    scale_size = int(img_size // crop_pct)
    eval_transforms = [
        transforms.Resize(scale_size, _pil_interp('bilinear')),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD),
        )
    ]
    eval_transforms = transforms.Compose(eval_transforms)
    return eval_transforms    

def visualize_timm_transforms(train_ds, idx, primary, secondary, final): 
    final = [tr for tr in final if type(tr) != transforms.Normalize]
    train_transforms_vis = transforms.Compose(primary+secondary+final)
    train_ds.transform = train_transforms_vis
    fig = plt.figure(figsize=(24, 24))
    for i in range(100): 
        np_img = train_ds[idx+i]['img'].numpy()
        _ = fig.add_subplot(10, 10, i+1)
        _ = plt.imshow(np_img.transpose(1, 2, 0))


