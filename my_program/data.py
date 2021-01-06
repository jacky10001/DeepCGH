# -*- coding: utf-8 -*-
"""
DeepCGH Datasets

Using tf.data.from_generator to Define Dataset
Add debug_generator and debug_sample

@author: Jacky Gao
Created on Wed Jan  6 19:40:07 2021
"""

import numpy as np
import tensorflow as tf
from skimage.draw import circle, line_aa


class DeepCGH_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, **params):
        try:
            assert params['object_type'] in ['Disk', 'Line', 'Dot'], 'Object type not supported'
            
            self.shape = params['shape']
            self.N = params['N']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.normalize = params['normalize']
        except:
            assert False, 'Not all parameters are provided!'
            
    def __get_line(self, shape, start, end):
        img = np.zeros(shape, dtype=np.float32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        img[rr, cc] = val * 1
        return img
    
    def get_circle(self, shape, radius, location):
        """Creates a single circle.
    
        Parameters
        ----------
        shape : tuple of ints
            Shape of the output image
        radius : int
            Radius of the circle.
        location : tuple of ints
            location (x,y) in the image
    
        Returns
        -------
        img
            a binary 2D image with a circle inside
        rr2, cc2
            the indices for a circle twice the size of the circle. This is will determine where we should not create circles
        """
        img = np.zeros(shape, dtype=np.float32)
        rr, cc = circle(location[0], location[1], radius, shape=img.shape)
        img[rr, cc] = 1
        # get the indices that are forbidden and return it
        rr2, cc2 = circle(location[0], location[1], 2*radius, shape=img.shape)
        return img, rr2, cc2

    def __get_allowables(self, allow_x, allow_y, forbid_x, forbid_y):
        '''
        Remove the coords in forbid_x and forbid_y from the sets of points in
        allow_x and allow_y.
        '''
        for i in forbid_x:
            try:
                allow_x.remove(i)
            except:
                continue
        for i in forbid_y:
            try:
                allow_y.remove(i)
            except:
                continue
        return allow_x, allow_y
    
    def __get_randomCenter(self, allow_x, allow_y):
        list_x = list(allow_x)
        list_y = list(allow_y)
        ind_x = np.random.randint(0,len(list_x))
        ind_y = np.random.randint(0,len(list_y))
        return list_x[ind_x], list_y[ind_y]
    
    def __get_randomStartEnd(self, shape):
        start = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        end = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        return start, end

    #% there shouldn't be any overlap between the two circles 
    def __get_RandDots(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random dots
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        xs = list(np.random.randint(0, shape[0], (n,)))
        ys = list(np.random.randint(0, shape[1], (n,)))
        
        for x, y in zip(xs, ys):
            image[x, y] = 1
            
        return image

    #% there shouldn't be any overlap between the two circles 
    def __get_RandLines(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random lines
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        for i in range(n):
            # generate centers
            start, end = self.__get_randomStartEnd(shape)
            
            # get circle
            img = self.__get_line(shape, start, end)
            image += img
        image -= image.min()
        image /= image.max()
        return image
    
    #% there shouldn't be any overlap between the two circles 
    def __get_RandBlobs(self, shape, maxnum = [10,12], radius = 5, intensity = 1):
        '''
        returns a single sample (2D image) with random blobs
        '''
        # random number of blobs to be generated
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        try: # in case the radius of the blobs is variable, get the largest diameter
            r = radius[-1]
        except:
            r = radius
        
        # define sets for storing the values
        allow_x = set(range(shape[0]))
        allow_y = set(range(shape[1]))
        if not self.centralized:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])))
        else:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])) + list(range(shape[0]//6, (5)*shape[0]//6)))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])) + list(range(shape[1]//6, (5)*shape[1]//6)))
        
        allow_x, allow_y = self.__get_allowables(allow_x, allow_y, forbid_x, forbid_y)
        count = 0
        # else
        for i in range(n):
            # generate centers
            x, y = self.__get_randomCenter(allow_x, allow_y)
            
            if isinstance(radius, list):
                r = int(np.random.randint(radius[0], radius[1]))
            else:
                r = radius
            
            if isinstance(intensity, list):
                int_4_this = int(np.random.randint(np.round(intensity[0]*100), np.round(intensity[1]*100)))
                int_4_this /= 100.
            else:
                int_4_this = intensity
            
            # get circle
            img, xs, ys = self.get_circle(shape, r, (x,y))
            allow_x, allow_y = self.__get_allowables(allow_x, allow_y, set(xs), set(ys))
            image += img * int_4_this
            count += 1
            if len(allow_x) == 0 or len(allow_y) == 0:
                break
        return image
    
    def coord2image(self, coords):
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            canvas = np.zeros(self.shape[:-1], dtype=np.float32)
        
            for i in range(coords.shape[-1]):
                img, _, __ = self.get_circle(self.shape[:-1], self.object_size, [coords[0, i], coords[1, i]])
                canvas += img.astype(np.float32)
            
            sample[:, :, plane] = (canvas>0)*1.
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
            
        sample -= sample.min()
        sample /= sample.max()
        
        return np.expand_dims(sample, axis = 0)
    
    #TODO
    def __make_sample(self):
        
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            if self.object_type == 'Disk':
                img = self.__get_RandBlobs(shape = (self.shape[0], self.shape[1]),
                                           maxnum = self.object_count,
                                           radius = self.object_size,
                                           intensity = self.intensity)
            elif self.object_type == 'Line':
                img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                           maxnum = self.object_count)
            elif self.object_type == 'Dot':
                img = self.__get_RandDots(shape = (self.shape[0], self.shape[1]),
                                          maxnum = self.object_count)
                

            sample[:, :, plane] = img
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
        
        sample -= sample.min()
        sample /= sample.max()
        
        return sample
    
    #TODO
    def debug_generator(self, batch_size):
        def _generator(N):
            for _ in range(N):
                sample = self.__make_sample()
                yield {'target':sample}, {'phi_slm':sample}
        
        data_types = tf.float64
        data_shape = self.shape
        types = {'target': data_types}, {'phi_slm': data_types}
        shapes = {'target': data_shape}, {'phi_slm': data_shape}
        return tf.data.Dataset.from_generator(
            _generator, args=[self.N],
            output_shapes=shapes, output_types=types).batch(batch_size)
    
    #TODO
    def debug_sample(self):
        sample = self.__make_sample()
        sample = np.expand_dims(sample, axis = 0)
        return sample