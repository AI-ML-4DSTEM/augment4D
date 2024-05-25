# Author: Joydeep Munshi
# Augmentation utility funciton for crystal4D
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
from augment4D.interpolate import dense_image_warp


class Image_Augmentation(object):
    def __init__(
        self,
        add_background=True,
        weightbackground=0.1,
        qBackgroundLorentz=0.1,
        add_shot=True,
        counts_per_pixel=1000,
        add_pattern_shift=True,
        xshift=1,
        yshift=1,
        add_ellipticity=True,
        ellipticity_scale=0.1,
        add_salt_and_pepper=None,
        salt_and_pepper_scale=1e-3,
        verbose=False,
        log_file="./logs/augment_log.csv",
        device="cpu",
    ):

        self.device = device.lower()
        if self.device == "gpu":
            self._xp = cp
            ## todo add option for setting GPU device
        else:
            self._xp = np

        self.set_params(
            add_background,
            weightbackground,
            qBackgroundLorentz,
            add_shot,
            counts_per_pixel,
            add_pattern_shift,
            xshift,
            yshift,
            add_ellipticity,
            ellipticity_scale,
            add_salt_and_pepper,
            salt_and_pepper_scale,
        )

        self.verbose = verbose
        self.log_file = log_file

        with open(self.log_file, "a") as f:
            f.write(
                "weighted_background,q_background_Lorentz,counts_per_pixel,xshift,yshift,ellipticity_scale,exx,eyy,exy,salt_and_pepper_scale\n"
            )

    def set_params(
        self,
        add_background=False,
        weightbackground=0.1,
        qBackgroundLorentz=0.1,
        add_shot=False,
        counts_per_pixel=1000,
        add_pattern_shift=False,
        xshift=1,
        yshift=1,
        add_ellipticity=False,
        ellipticity_scale=0.1,
        add_salt_and_pepper=None,
        salt_and_pepper_scale=1e-3,
    ):
        xp = self._xp
        self.add_background = add_background
        self.weightbackground = weightbackground if self.add_background else 0
        self.qBackgroundLorentz = qBackgroundLorentz if self.add_background else 0

        self.add_shot = add_shot
        self.counts_per_pixel = counts_per_pixel if self.add_shot else self._xp.inf

        self.add_pattern_shift = add_pattern_shift
        self.xshift = xshift if self.add_pattern_shift else 0
        self.yshift = yshift if self.add_pattern_shift else 0

        self.add_ellipticity = add_ellipticity
        if add_ellipticity:
            self.ellipticity_scale = ellipticity_scale
            self.exx = xp.random.normal(loc=1, scale=self.ellipticity_scale)
            self.eyy = xp.random.normal(loc=1, scale=self.ellipticity_scale)
            self.exy = xp.random.normal(loc=0, scale=self.ellipticity_scale)
        else:
            self.ellipticity_scale = 0
            self.exx = 1
            self.eyy = 1
            self.exy = 0

        self.add_salt_and_pepper = add_salt_and_pepper
        self.salt_and_pepper_scale = (
            salt_and_pepper_scale if self.add_salt_and_pepper else 0
        )

    def get_params(self):
        print("Printing augmentation summary... \n", end="\r")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n", end="\r")
        print(f"Weighted background: {self.weightbackground}")
        print(f"Background plasmon: {self.qBackgroundLorentz}")
        print(f"Counts per pixel: {self.counts_per_pixel:.1e}")
        print(f"Pattern shift: {self.xshift},{self.yshift}")
        print(f"Ellipticity scaling: {self.ellipticity_scale}")
        print(f"Ellipticity params (exx, eyy, exy): ({self.exx:.2f}, {self.eyy:.2f}, {self.exy:.2f})")
        print(f"Salt & pepper scaling: {self.salt_and_pepper_scale:.1e}")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n", end="\r")

    def write_logs(self):
        with open(self.log_file, "a") as f:
            f.write(
                f"{self.weightbackground},{self.qBackgroundLorentz},{self.counts_per_pixel},{self.xshift},{self.yshift},{self.ellipticity_scale},{self.exx},{self.eyy},{self.exy},{self.salt_and_pepper_scale}\n"
            )

    def _as_array(self, ar):
        return self._xp.asarray(ar)

    def _get_qx_qy(self, input_shape, pixel_size_AA=0.20):
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
        xp = self._xp
        if hasattr(pixelSize, "__len__"):
            assert (
                len(pixelSize) == 2
            ), "pixelSize must either be a scalar or have length 2"
            pixelSize_x = pixelSize[0]
            pixelSize_y = pixelSize[1]
        else:
            pixelSize_x = pixelSize
            pixelSize_y = pixelSize

        qx = xp.fft.fftfreq(Nx, pixelSize_x)
        qy = xp.fft.fftfreq(Ny, pixelSize_y)
        qy, qx = xp.meshgrid(qy, qx)
        return qx, qy

    def augment_img(self, inputs, probe=None):
        start_time = time.time()
        input_shape = inputs.shape
        if self.add_background:
            input_noise = self.background_plasmon(inputs, probe)
        else:
            input_noise = inputs
            self.weightbackground = 0
            self.qBackgroundLorentz = 0

        if self.add_shot:
            input_noise = self.shot_noise(input_noise)
        else:
            self.counts_per_pixel = self._xp.inf

        if self.add_ellipticity or self.add_pattern_shift:
            input_noise = self.elastic_distort(input_noise)
            if not self.add_ellipticity:
                self.ellipticity_scale = 0
            elif not self.add_pattern_shift:
                self.xshift = self.yshift = 0
        else:
            self.ellipticity_scale = 0
            self.xshift = self.yshift = 0

        if self.add_salt_and_pepper:
            input_noise = self.apply_salt_and_pepper(input_noise)
        else:
            self.salt_and_pepper_scale = 0

        t = time.time() - start_time

        if self.verbose:
            self.get_params()
            self.write_logs()
            print(
                f"Augmentation Status: it took {t/60:.1e} minutes to augment {input_shape[0]} images..."
            )
        else:
            self.write_logs()

        return input_noise

    def shot_noise(self, inputs):
        """
        Apply Shot noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        image_noise = xp.random.poisson(
            lam=xp.maximum(image, 0) * self.counts_per_pixel
        ) / float(self.counts_per_pixel)

        return image_noise

    def elastic_distort(self, inputs):
        """
        Elliptic distortion and bilinear interpolated x/y shifts
        """
        image = self._as_array(inputs)
        batch_size, height, width, _channels = image.shape
        xp = self._xp

        # original -- was hardcoded for some reason
        # exx = 0.7 * scale
        # eyy = -0.5 * scale
        # exy = -0.9 * scale

        m = [[1 + self.exx, self.exy / 2], [self.exy / 2, 1 + self.eyy]]

        grid_x, grid_y = xp.meshgrid(xp.arange(width), xp.arange(height))
        grid_x = (grid_x - grid_x.mean()).astype(xp.float32)
        grid_y = (grid_y - grid_y.mean()).astype(xp.float32)

        if self.add_pattern_shift:
            grid_x += self.xshift
            grid_y += self.yshift

        flow_x = grid_x * m[0][0] + grid_y * m[1][0]
        flow_y = grid_x * m[0][1] + grid_y * m[1][1]

        flow_grid = xp.stack([flow_y, flow_x], axis=2).astype(xp.float32)
        batched_flow_grid = xp.expand_dims(flow_grid, axis=0)

        batched_flow_grid = xp.repeat(batched_flow_grid, batch_size, axis=0)
        assert (
            batched_flow_grid.shape[0] == batch_size
        ), f"The flow batch size ({batched_flow_grid.shape[0]}) != image batch size ({batch_size})"

        imageOut = dense_image_warp(image, batched_flow_grid)

        return imageOut

    def background_plasmon(self, inputs, probe=None):
        """
        Apply background plasmon noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        batch_size, height, width, channels = image.shape

        qx, qy = self._get_qx_qy([height, width])
        CBEDbg = 1.0 / (qx**2 + qy**2 + self.qBackgroundLorentz**2)
        CBEDbg = xp.squeeze(CBEDbg)
        CBEDbg = CBEDbg / xp.sum(CBEDbg)

        CBEDbg = xp.expand_dims(CBEDbg, axis=0)
        CBEDbg = xp.repeat(CBEDbg, batch_size, axis=0)
        CBEDbg = xp.expand_dims(CBEDbg, axis=-1)
        CBEDbg = xp.repeat(CBEDbg, channels, axis=0)

        CBEDbg = xp.transpose(CBEDbg, (3, 0, 1, 2))
        probe = xp.transpose(probe, (3, 0, 1, 2))

        CBEDbg_ff = xp.fft.fft2(CBEDbg).astype(xp.complex128)
        probe_ff = xp.fft.fft2(probe).astype(xp.complex128)
        mul_ff = CBEDbg_ff * probe_ff

        CBEDbgConv = xp.fft.fftshift(xp.fft.ifft2(mul_ff), axes=[2, 3])
        CBEDbgConv = xp.transpose(CBEDbgConv, (1, 2, 3, 0))

        CBEDout = (
            inputs.astype(xp.float32) * (1 - self.weightbackground)
            + CBEDbgConv.real * self.weightbackground
        )

        return CBEDout

    def fourrier_shift_ar(self, inputs):
        """
        Apply pixel shift to the pattern using Fourier shift theorem for subpixel shifting
        this can add ringing due to undersampling
        """
        image = self._as_array(inputs)
        _batch_size, height, width, _channels = image.shape
        xp = self._xp

        qx, qy = self._make_fourier_coord(height, width, 1)
        ar = xp.transpose(image, (0, 3, 1, 2)).astype(xp.complex64)

        w = xp.exp(-(2.0j * xp.pi) * ((self.yshift * qy) + (self.xshift * qx)))
        shifted_ar = xp.fft.ifft2(xp.fft.fft2(ar) * w).real

        shifted_ar = xp.transpose(shifted_ar, (0, 2, 3, 1))
        return shifted_ar

    def apply_salt_and_pepper(self, inputs):
        minval = inputs.min()
        imfac = (inputs - minval).max()
        im_sp = self.get_salt_and_pepper(inputs, self.salt_and_pepper_scale)
        im_sp = (im_sp * imfac) + minval
        return im_sp

    def get_salt_and_pepper(self, image, amount=1e-3, salt_vs_pepper=1.0, low_clip=0):
        """
        expects normalized input [0,1]
        based off skimage implementation
        """
        xp = self._xp
        out = image.copy()
        flipped = xp.random.random(out.shape) <= amount
        salted = xp.random.random(out.shape) <= salt_vs_pepper
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = low_clip
        return out
