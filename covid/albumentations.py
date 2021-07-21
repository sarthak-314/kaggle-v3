from albumentations import (
    Compose, OneOf, RandomResizedCrop, Resize, HorizontalFlip, IAAPerspective, ShiftScaleRotate, 
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, 
    HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Cutout, 
    IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, 
    RandomGamma, ElasticTransform, ChannelShuffle, RGBShift,Rotate, Normalize, 
    RandomSizedCrop, VerticalFlip, RandomBrightnessContrast
)
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.pytorch.transforms import ToTensorV2
import albumentations




def get_resize_fn(img_size): 
    return RandomResizedCrop(img_size, img_size, scale=(0.5, 1), ratio=(0.75, 1.33), always_apply=True)

def get_pad_img(img_size): 
    PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True)

def get_soft_augmentations(img_size): 
    limit = (-0.2, 0.2) # Brightness and contrast limit
    rotation_angle = 120
    return [
        get_pad_img(img_size), get_resize_fn(img_size), 
        RandomRotate90(), Flip(), Rotate(limit=rotation_angle, p=0.5),
        RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=0.5), 
        Cutout(num_holes=4, max_h_size=img_size//16, max_w_size=img_size//16, fill_value=0, p=0.5), 
        Cutout(num_holes=8, max_h_size=img_size//32, max_w_size=img_size//32, fill_value=1, p=0.5), 
        Normalize(), 
    ]
    
def get_hard_augmentations(img_size): 
    return [
        get_pad_img(img_size), get_resize_fn(img_size), 
        RandomRotate90(), Flip(), Transpose(),
        OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
        Cutout(num_holes=4, max_h_size=img_size//16, max_w_size=img_size//16, fill_value=0, p=0.5), 
        Cutout(num_holes=8, max_h_size=img_size//32, max_w_size=img_size//32, fill_value=1, p=0.5), 
        OneOf([ MotionBlur(p=.2), MedianBlur(blur_limit=3, p=.1), Blur(blur_limit=3, p=.1) ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([ OpticalDistortion(p=0.3), GridDistortion(p=.1), IAAPiecewiseAffine(p=0.3) ], p=0.2),
        OneOf([ CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(), RandomContrast(), RandomBrightness()], p=0.3),
        Normalize(), 
    ]
    
    
def get_test_augmentations(img_size):
    return [
        Resize(img_size, img_size), 
        Normalize(), 
    ]


def get_train_transforms(img_size, aug_hard=False, to_tensor=False):
    augmentations = get_hard_augmentations(img_size) if aug_hard else get_soft_augmentations(img_size)
    if to_tensor: 
        augmentations.append(ToTensorV2(p=1.0))
    train_transforms = Compose(augmentations)
    return train_transforms

def get_test_transforms(img_size, to_tensor=False): 
    augmentations = get_test_augmentations(img_size)
    if to_tensor: 
        augmentations.append(ToTensorV2(p=1.0))
    test_transforms = Compose(augmentations)
    return test_transforms
    