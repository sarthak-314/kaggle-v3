import tensorflow as tf
from utils.tf_aug_dump import *

# TEMP CONFIG FOR THE RUN
CHANNELS = 3
IMG_SIZE = 512
BATCH_SIZE = 64
def train_img_augment(img, label):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)    
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shift = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_cutmix = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_mixup = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    # Flips
    if p_spatial >= .2:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        
    # Rotates
    if p_rotate > .75:
        img = tf.image.rot90(img, k=3) # rotate 270º
    elif p_rotate > .5:
        img = tf.image.rot90(img, k=2) # rotate 180º
    elif p_rotate > .25:
        img = tf.image.rot90(img, k=1) # rotate 90º
    
    if p_rotation >= .3: # Rotation
        img = transform_rotation(img, height=IMG_SIZE, rotation=45.)
    if p_shift >= .3: # Shift
        img = transform_shift(img, height=IMG_SIZE, h_shift=15., w_shift=15.)
    if p_shear >= .3: # Shear
        img = transform_shear(img, height=IMG_SIZE, shear=20.)
        
    # Crops
    if p_crop > .4:
        crop_size = tf.random.uniform([], int(IMG_SIZE*.7), IMG_SIZE, dtype=tf.int32)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, CHANNELS])
    elif p_crop > .7:
        if p_crop > .9:
            img = tf.image.central_crop(img, central_fraction=.7)
        elif p_crop > .8:
            img = tf.image.central_crop(img, central_fraction=.8)
        else:
            img = tf.image.central_crop(img, central_fraction=.9)
            
    img = tf.image.resize(img, size=[IMG_SIZE, IMG_SIZE])
        
    # Pixel-level transforms
    if p_pixel >= .2:
        if p_pixel >= .8:
            img = tf.image.random_saturation(img, lower=0, upper=2)
        elif p_pixel >= .6:
            img = tf.image.random_contrast(img, lower=.8, upper=2)
        elif p_pixel >= .4:
            img = tf.image.random_brightness(img, max_delta=.2)
        else:
            img = tf.image.adjust_gamma(img, gamma=.6)

    # Cutmix & Mixup
    img, label = cutmix(img, label, p_cutmix)
    img, label = mixup(img, label, p_mixup)
    
    return img, label


def cutmix(image, label, PROBABILITY = 1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    CLASSES = 104
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,IMG_SIZE),tf.int32)
        y = tf.cast( tf.random.uniform([],0,IMG_SIZE),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( IMG_SIZE * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(IMG_SIZE,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(IMG_SIZE,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:IMG_SIZE,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:IMG_SIZE,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/IMG_SIZE/IMG_SIZE,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,IMG_SIZE,IMG_SIZE,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,CLASSES))
    return image2,label2


def mixup(image, label, PROBABILITY = 1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    CLASSES = 104
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,IMG_SIZE,IMG_SIZE,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,CLASSES))
    return image2,label2



