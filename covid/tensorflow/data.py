import tensorflow as tf 
import numpy as np


class KaggleGenerator(tf.keras.utils.Sequence): 
    def __init__(self, )
class Covid19Generator(tf.keras.utils.Sequence):
    def __init__(self, img_path, msk_path, data, batch_size, random_state, 
                 idim, mdim, shuffle=True, transform=None, is_train=False):
        self.idim = idim
        self.mdim = mdim  
        self.data = data
        self.shuffle  = shuffle
        self.random_state = random_state
        
        self.img_path = img_path
        self.msk_path = msk_path
        self.is_train = is_train
        
        self.augment  = transform
        self.batch_size = batch_size
        
        self.list_idx = data.index.values
        self.label = self.data[['Negative for Pneumonia', 
                                'Typical Appearance', 
                                'Indeterminate Appearance', 
                                'Atypical Appearance']] if self.is_train else np.nan
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_idx) / self.batch_size))
    
    def __getitem__(self, index):
        batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        idx = [self.list_idx[k] for k in batch_idx]
        
        Data = np.zeros((self.batch_size,) + self.idim + (3,), dtype="float32")
        Mask = np.zeros((self.batch_size,) + self.mdim + (1,), dtype="float32")
        Target = np.zeros((self.batch_size, 4), dtype = np.float32)

        for i, k in enumerate(idx):
            # load the image file using cv2
            image = cv2.imread(self.img_path + self.data['id'][k] + '.png')[:, :, [2, 1, 0]]
            mask = cv2.imread(self.msk_path + self.data['id'][k] + '.png', 0)
            
            try:
                mask = cv2.resize(mask, self.mdim)[:, :, np.newaxis]
            except:
                mask = np.zeros_like(cv2.resize(image[:,:,:1], self.mdim))[:, :, np.newaxis]
          
            res = self.augment(image=image)
            image = res['image']
            
            # mask normalization must
            mask = mask.astype(np.float32)/255.0 

            # assign 
            if self.is_train:
                Data[i,] = image
                Mask[i,] = mask
                Target[i,] = self.label.iloc[k,].values #.values
            else:
                Data[i,] =  image 
        
        inps = {'input': Data}
        outs = {'clss': Target, 'segg': Mask}
        return inps, outs
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_idx))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indices)
