from tensorflow.keras import backend as K  
import tensorflow as tf 
import math 

# data augmentation @cdeotte kernel: https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
def transform_rotation(image, height, rotation):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated
    DIM = height
    XDIM = DIM%2 #fix for size 331
    
    rotation = rotation * tf.random.uniform([1],dtype='float32')
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3])

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(rotation_matrix,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

def transform_shear(image, height, shear):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly sheared
    DIM = height
    XDIM = DIM%2 #fix for size 331
    
    shear = shear * tf.random.uniform([1],dtype='float32')
    shear = math.pi * shear / 180.
        
    # SHEAR MATRIX
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3])    

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(shear_matrix,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

def transform_shift(image, height, h_shift, w_shift):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly shifted
    DIM = height
    XDIM = DIM%2 #fix for size 331
    
    height_shift = h_shift * tf.random.uniform([1],dtype='float32') 
    width_shift = w_shift * tf.random.uniform([1],dtype='float32') 
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
        
    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3])

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(shift_matrix,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

def train_img_augment(img, img_size, channels):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)    
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shift = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
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
        img = transform_rotation(img, height=img_size, rotation=45.)
    if p_shift >= .3: # Shift
        img = transform_shift(img, height=img_size, h_shift=15., w_shift=15.)
    if p_shear >= .3: # Shear
        img = transform_shear(img, height=img_size, shear=20.)
        
    # Crops
    if p_crop > .4:
        crop_size = tf.random.uniform([], int(img_size*.7), img_size, dtype=tf.int32)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, channels])
    elif p_crop > .7:
        if p_crop > .9:
            img = tf.image.central_crop(img, central_fraction=.7)
        elif p_crop > .8:
            img = tf.image.central_crop(img, central_fraction=.8)
        else:
            img = tf.image.central_crop(img, central_fraction=.9)
            
    img = tf.image.resize(img, size=[img_size, img_size])
        
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

    return img

def cutmix(image, label, batch_size, img_size, classes=4, prob = 1.0):
    label = tf.cast(label, tf.float32)
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    imgs = []; labs = []
    for j in range(batch_size):
        # DO CUTMIX WITH prob DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<= prob, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,batch_size),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,img_size),tf.int32)
        y = tf.cast( tf.random.uniform([],0,img_size),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( img_size * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(img_size,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(img_size,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:img_size,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:img_size,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/img_size/img_size,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],classes)
            lab2 = tf.one_hot(label[k],classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(batch_size,img_size,img_size,3))
    label2 = tf.reshape(tf.stack(labs),(batch_size,classes))
    print('image2: ', image2)
    print('label2: ', label2)
    return image2,label2


def mixup(image, label, batch_size, img_size, classes, prob = 1.0):
    label = tf.cast(label, tf.float32)
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    imgs = []; labs = []
    for j in range(batch_size):
        # DO MIXUP WITH prob DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=prob, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,batch_size),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],classes)
            lab2 = tf.one_hot(label[k],classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(batch_size,img_size,img_size,3))
    label2 = tf.reshape(tf.stack(labs),(batch_size,classes))
    return image2,label2


def get_train_transforms(img_size, channels): 
    train_transforms_fn = lambda img: train_img_augment(img, img_size, channels)
    return train_transforms_fn

def get_batch_transforms(img_size, batch_size, classes, prob=0.5):
    def batch_transforms_fn(img, label): 
        img, label = cutmix(img, label, batch_size, img_size, classes, prob=prob)
        img, label = mixup(img, label, batch_size, img_size, classes, prob=prob)
        return img, label 
    return batch_transforms_fn    


def get_eval_transforms(img_size, channels): 
    def resize_fn(img): 
        img = tf.image.resize(img, size=[img_size, img_size])
        return img
    return resize_fn