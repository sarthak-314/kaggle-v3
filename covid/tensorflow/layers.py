

import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np


class BiFPN(layers.Layer): 
    """
    From EfficientDet Paper
    Args: 
        in_channels: list of feature size of each layer from backbone with [(width, channel)]
    """
    def __init__(self, in_channels): 
        super(BiFPN, self).__init__()
        self.epsilon = 1e-4
        self.num_input_layers = len(in_channels)
        in_wd, in_ch = zip(*in_channels)
        
        self.td_weights = []
        self.out_weights = []
        self.td_convs = []
        self.out_convs = []

        self.out_weights.append(tf.random.normal([3]))
        self.out_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[0], 3, padding='same'),
                                                    layers.BatchNormalization()]))
        for i in range(self.num_input_layers-2):
            self.td_weights.append(tf.random.normal([2]))
            self.td_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[i+1], 3, padding='same'),
                                                        layers.BatchNormalization()]))
            self.out_weights.append(tf.random.normal([3]))
            self.out_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[i+1], 3, padding='same'),
                                                        layers.BatchNormalization()]))
        self.td_weights.append(tf.random.normal([2]))
        self.td_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[-1], 3, padding='same'),
                                                    layers.BatchNormalization()]))

        self.upconvs  = [tf.keras.Sequential([layers.UpSampling2D(u),
                                            layers.Conv2D(c,k,padding=pad)])
                                            for u,c,k,pad in zip([2,2,2,2],
                                                                in_ch[1:],
                                                                [2,3,2,3],
                                                                ['valid','same','valid','same'])]
        self.downconvs= [tf.keras.Sequential([layers.ZeroPadding2D(pad),
                                            layers.AveragePooling2D(p),
                                            layers.Conv2D(c,3,padding='same')])
                                            for c,p,pad in zip(in_ch[:-1],
                                                                [2,2,2,2],
                                                                [1,0,1,0])]


    def call(self, xs):
        '''
        Args:
            xs: (Variable) list of features of each layer
        e.g.
        in block 1,2,4,7,14 in MobileNetV2 is used,
        shape of xs: [[?,10,10,160],[?,19,19,64],[?,38,38,32],[?,75,75,24],[?,150,150,32]]
        '''
        tds = [xs[0]]
        for i in range(self.num_input_layers-1):
            tds.append(self.td_convs[i](
                (self.td_weights[i][0]*xs[i+1]
                + self.td_weights[i][1]*self.upconvs[i](tds[-1]))
                / (tf.math.reduce_sum(self.td_weights[i])+self.epsilon)
            ))

        outs = [tds[-1]]
        for i in range(self.num_input_layers-2,0,-1):
            outs.append(self.out_convs[i](
                (self.out_weights[i][0]*xs[i]
                + self.out_weights[i][1]*tds[i]
                + self.out_weights[i][2]*self.downconvs[i](outs[-1]))
                / (tf.math.reduce_sum(self.out_weights[i])+self.epsilon)
            ))
        outs.append(self.out_convs[0](
            (self.out_weights[0][0]*xs[0]
            + self.out_weights[0][1]*self.downconvs[0](outs[-1]))
            / (tf.math.reduce_sum(self.out_weights[0])+self.epsilon)
        ))

        return outs