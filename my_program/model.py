# -*- coding: utf-8 -*-
"""
Define Unet of DeepCGH
eturn KERAS Functional Model

@author: Jacky Gao
Created on Wed Jan  6 20:54:54 2021
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D


def unet(shape, n_kernels, IF):
    def interleave(x):
        return tf.nn.space_to_depth(input=x, block_size=IF, data_format='NHWC')
            
    def deinterleave(x):
        return tf.nn.depth_to_space(input=x, block_size=IF, data_format='NHWC')
            
    def __ifft_AmPh(x):
        ''' Input is Amp x[1] and Phase x[0]. Spits out the angle of ifft. '''
        img = tf.dtypes.complex(tf.squeeze(x[0], axis=-1), 0.) * tf.math.exp(tf.dtypes.complex(0., tf.squeeze(x[1], axis=-1)))
        img = tf.signal.ifftshift(img, axes = [1, 2])
        fft = tf.signal.ifft2d(img)
        phase = tf.expand_dims(tf.math.angle(fft), axis=-1)
        return phase
            
    def __cbn(ten, n_kernels, act_func):
        x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        return x1 
    
    def __cc(ten, n_kernels, act_func):
        x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
        x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
        return x1
    
    inp = Input(shape=shape, name='target')
    act_func = 'relu'
    x1_1 = Lambda(interleave, name='Interleave')(inp)
    # Block 1
    x1 = __cbn(x1_1, n_kernels[0], act_func)
    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    # Block 2
    x2 = __cbn(x2, n_kernels[1], act_func)
    encoded = MaxPooling2D((2, 2), padding='same')(x2)
    # Bottleneck
    encoded = __cc(encoded, n_kernels[2], act_func)
    #
    x3 = UpSampling2D(2)(encoded)
    x3 = Concatenate()([x3, x2])
    x3 = __cc(x3, n_kernels[1], act_func)
    #
    x4 = UpSampling2D(2)(x3)
    x4 = Concatenate()([x4, x1])
    x4 = __cc(x4, n_kernels[0], act_func)
    #
    x4 = __cc(x4, n_kernels[1], act_func)
    x4 = Concatenate()([x4, x1_1])
    #
    phi_0_ = Conv2D(IF**2, (3, 3), activation=None, padding='same')(x4)
    phi_0 = Lambda(deinterleave, name='phi_0')(phi_0_)
    amp_0_ = Conv2D(IF**2, (3, 3), activation='relu', padding='same')(x4)
    amp_0 = Lambda(deinterleave, name='amp_0')(amp_0_)
    
    phi_slm = Lambda(__ifft_AmPh, name='phi_slm')([amp_0, phi_0])
    
    return Model(inp, phi_slm)