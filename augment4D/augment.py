#Author: Joydeep Munshi
#Augmentation utility funciton for crystal4D
"""
This is augmentation pipeline for 4DSTEM/STEM diffraction (CBED) images.
The different available augmentation includes elliptic distrotion, plasmonic background
noise and poisson shot noise. The elliptic distortion is recommeded to be applied on
CBED, Probe, Potential Vg and Qz tilts while other noise are only applied on CBED input.
"""
import time
import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

import scipy.signal as sp
import numpy.matlib as nm
from itertools import product
# from augment4D.interpolate import dense_image_warp

class Image_Augmentation(object):
    def __init__(self,
                 background = False,
                 shot =False,
                 pattern_shift = False,
                 ellipticity = False,
                 counts_per_pixel = 1000,
                 qBackgroundLorentz = 0.1,
                 weightbackground = 0.1,
                 scale = 1.0,
                 xshift = 1,
                 yshift = 1,
                 verbose = False,
                 log_file = './logs/augment_log.csv',
                 device='cpu'):
        self.background = background
        self.shot = shot
        self.pattern_shift = pattern_shift
        self.ellipticity = ellipticity
        self.device = device.lower()
        if self.device == "gpu":
            self._xp = cp
            ## set GPU device
        else:
            self._xp = np

        if self.background:
            print('background')
            self.weightbackground = weightbackground
            self.qBackgroundLorentz = qBackgroundLorentz
        if self.shot:
            self.counts_per_pixel = counts_per_pixel
        if self.pattern_shift:
            self.xshift = xshift
            self.yshift = yshift
        else:
            self.xshift = 0
            self.yshift = 0
        if self.ellipticity:
            self.scale = scale

        self.verbose = verbose
        self.log_file = log_file

        file_object = open(self.log_file, 'a')
        file_object.write('ellipticity,shot,background,pattern_shift \n')
        file_object.close()

    def set_params(self,
                 background = True,
                 shot =True,
                 pattern_shift = True,
                 ellipticity = True,
                 counts_per_pixel = 1000,
                 qBackgroundLorentz = 0.1,
                 weightbackground = 0.1,
                 scale = 1.0,
                 xshift = 1,
                 yshift = 1):

        self.background = background
        self.shot = shot
        self.pattern_shift = pattern_shift
        self.ellipticity = ellipticity

        if self.background:
            self.weightbackground = weightbackground
            self.qBackgroundLorentz = qBackgroundLorentz
        if self.shot:
            self.counts_per_pixel = counts_per_pixel
        if self.pattern_shift:
            self.xshift = xshift
            self.yshift = yshift
        if self.ellipticity:
            self.scale = scale

    def get_params(self):
        print('Printing augmentation summary... \n',end = "\r")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n',end = "\r")
        print('Shots per pixel: {} \n'.format(self.counts_per_pixel),end = "\r")
        print('Background plasmon: {} \n'.format(self.qBackgroundLorentz),end = "\r")
        print('Ellipticity scaling: {} \n'.format(self.scale),end = "\r")
        print('Pattern shift: {},{} \n'.format(self.xshift,self.yshift),end = "\r")
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n', end = "\r")


    def write_logs(self):
        file_object = open(self.log_file, 'a')
        file_object.write('{},{},{},{} \n'.format(self.ellipticity,self.shot,self.background,self.pattern_shift))
        file_object.close()

    def _as_array(self, ar):
        return self._xp.asarray(ar)


    # def _get_dim(self, x, idx):
    #     if x.shape.ndims is None:
    #         return tf.shape(x)[idx]
    #     return x.shape[idx] or tf.shape(x)[idx]


    def _get_qx_qy(self, input_shape, pixel_size_AA = 0.20):
        """
        get qx,qy from cbed
        """
        N = input_shape
        qx = np.sort(np.fft.fftfreq(N[0], pixel_size_AA)).reshape((N[0], 1, 1))
        qy = np.sort(np.fft.fftfreq(N[1], pixel_size_AA)).reshape((1, N[1], 1))

        return qx, qy

    def _make_fourier_coord(self, Nx, Ny, pixelSize):
        """
        Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
        """
        if hasattr(pixelSize, '__len__'):
            assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
            pixelSize_x = pixelSize[0]
            pixelSize_y = pixelSize[1]
        else:
            pixelSize_x = pixelSize
            pixelSize_y = pixelSize

        qx = np.fft.fftfreq(Nx, pixelSize_x)
        qy = np.fft.fftfreq(Ny, pixelSize_y)
        qy, qx = np.meshgrid(qy, qx)
        return qx, qy

    def augment_img(self, inputs, probe = None):
        start_time = time.time()
        input_shape = inputs.shape
        if self.background:
            input_noise = self.background_plasmon(inputs, probe)
        else:
            input_noise = inputs
            self.weightbackground = 0
            self.qBackgroundLorentz = 0

        if self.shot:
            input_noise = self.shot_noise(input_noise)
        else:
            self.counts_per_pixel = 0

        ################
        if self.pattern_shift:
            input_noise = self.pattern_shift_ar(input_noise)
        else:
            self.xshift = 0
            self.yshift = 0

        if self.ellipticity:
            input_noise = self.elliptic_distort(input_noise)
        else:
            self.scale = 0

        t = time.time() - start_time
        t = t/60

        if self.verbose:
            self.get_params()
            self.write_logs()
            print('Augmentation Status: it took {} minutes to augment {} images... \n'.format(t, input_shape[0]), end = "\r")
        else:
            self.write_logs()

        return input_noise

    # def scale_image(self, inputs): ### unused
    #     '''
    #     Scale image between 0 and 1
    #     '''
    #     input_shape = tf.shape(inputs)
    #     mean = tf.squeeze(tf.math.reduce_sum(inputs, axis = (1,2)))
    #     inputs_scaled = tf.transpose(tf.transpose(inputs, [3,1,2,0])/mean, [3,1,2,0])

    #     return inputs_scaled

    def shot_noise(self, inputs):
        """
        Apply Shot noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        ###image_scale = self.scale_image(inputs)
        image_noise = xp.random.poisson(shape = [], lam = xp.maximum(image,0) * self.counts_per_pixel)/float(self.counts_per_pixel)

        return image_noise

    # def elliptic_distort(
    #     self, inputs):
    #     """
    #     Elliptic distortion
    #     """
    #     image = self._as_array(inputs)
    #     batch_size, height, width, channels = (
    #         self._get_dim(image, 0),
    #         self._get_dim(image, 1),
    #         self._get_dim(image, 2),
    #         self._get_dim(image, 3),
    #     )

    #     exx = 0.7 * self.scale
    #     eyy = -0.5 * self.scale
    #     exy = -0.9 * self.scale

    #     m = [[1+exx, exy/2], [exy/2, 1+eyy]]

    #     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    #     grid_x = grid_x - tf.math.reduce_mean(grid_x)
    #     grid_y = grid_y - tf.math.reduce_mean(grid_y)

    #     grid_x = tf.cast(grid_x, tf.float32)
    #     grid_y = tf.cast(grid_y, tf.float32)

    #     #print(stacked_grid)

    #     flow_x =  (grid_x*m[0][0] + grid_y*m[1][0])
    #     flow_y =  (grid_x*m[0][1] + grid_y*m[1][1])

    #     flow_grid = tf.cast(tf.stack([flow_y, flow_x], axis=2), grid_x.dtype)
    #     batched_flow_grid = tf.expand_dims(flow_grid, axis=0)

    #     batched_flow_grid = tf.repeat(batched_flow_grid, batch_size, axis=0)
    #     #assert(self._get_dim(batched_flow_grid, 0) == batch_size); "The flow batch size and the image batch size is different!!"

    #     # imageOut = dense_image_warp(image, batched_flow_grid)

    #     return imageOut

    def background_plasmon(self, inputs, probe):
        """
        Apply background plasmon noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        batch_size, height, width, channels = image.shape

        qx,qy = self._get_qx_qy([height,width])
        CBEDbg = 1./ (qx**2 + qy**2 + self.qBackgroundLorentz**2)
        CBEDbg = xp.squeeze(CBEDbg)
        CBEDbg = CBEDbg / xp.sum(CBEDbg)

        CBEDbg = xp.expand_dims(CBEDbg, axis=0)
        CBEDbg = xp.repeat(CBEDbg, batch_size, axis=0)
        CBEDbg = xp.expand_dims(CBEDbg, axis=-1)
        CBEDbg = xp.repeat(CBEDbg, channels, axis=0)

        CBEDbg = xp.transpose(CBEDbg, (3,0,1,2))
        probe = xp.transpose(probe, (3,0,1,2))

        CBEDbg_ff = xp.fft.fft2(CBEDbg).astype(xp.complex128)
        probe_ff = xp.fft.fft2(probe).astype(xp.complex128)
        mul_ff = CBEDbg_ff * probe_ff


        CBEDbgConv = xp.fft.fftshift(xp.fft.ifft2(mul_ff), axes = [2,3])
        CBEDbgConv = xp.transpose(CBEDbgConv, (1,2,3,0))

        CBEDout = inputs.astype(xp.float32) * (1-self.weightbackground) + CBEDbgConv.astype(xp.float32) * self.weightbackground

        return CBEDout

    def pattern_shift_ar(self, inputs):
        """
        Apply pixel shift to the pattern using Fourier shift theorem for subpixel shifting
        """
        image = self._as_array(inputs)
        _batch_size, height, width, _channels = image.shape
        xp = self._xp

        qx, qy = self._make_fourier_coord(height, width, 1)

        ar = xp.transpose(image, (0,3,1,2)).astype(xp.complex64)

        w = xp.exp(-(2j * xp.pi) * ((self.yshift * qy) + (self.xshift * qx)))
        shifted_ar = xp.fft.ifft2(xp.fft.fft2(ar) * w).real

        shifted_ar = xp.transpose(shifted_ar, (0,2,3,1))

        return shifted_ar