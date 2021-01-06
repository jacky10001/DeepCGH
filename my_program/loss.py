# -*- coding: utf-8 -*-
"""
Custom Loss Function
Using subclass to build loss function

@author: Jacky Gao
Created on Wed Jan  6 20:53:44 2021
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


class CustomLoss(Loss):
    def __init__(self, shape, zs, wavelength, ps, name="custom_loss"):
        super().__init__(name=name)
        self.zs = zs
        self.shape = shape
        self.wavelength = wavelength
        self.ps = ps
        
    def get_H(self, zs, shape, lambda_, ps):
        Hs = []
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                                   np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))
            fx = x/ps/shape[0]
            fy = y/ps/shape[1]
            exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
            Hs.append(exp.astype(np.complex64))
        return Hs
            
    def phi_slm(self, phi_slm):
        i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
        return tf.math.exp(i_phi_slm)
        
    def prop(self, cf_slm, H=None, center=False):
        if not center:
            H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
            cf_slm *= tf.signal.fftshift(H, axes = [1, 2])
        fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
        img = tf.cast(tf.expand_dims(tf.abs(tf.pow(fft, 2)), axis=-1), dtype=tf.dtypes.float32)
        return img
                
    def accuracy(self, y_true, y_pred):
        denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
        return 1-tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])+1)/(denom+1), axis = 0)
        
    def call(self, y_true, y_pred):
        zs = self.zs
        shape = self.shape
        wavelength = self.wavelength
        ps = self.ps
        
        frames = []
        Hs = self.get_H(zs, shape, wavelength, ps)
        cf_slm = self.phi_slm(y_pred)
        for H, z in zip(Hs, zs):
            frames.append(self.prop(cf_slm, tf.keras.backend.constant(H, dtype = tf.complex64)))
        frames.insert(shape[-1] // 2, self.prop(cf_slm, center = True))
        y_pred = tf.concat(values=frames, axis = -1)
        return self.accuracy(y_true, y_pred)