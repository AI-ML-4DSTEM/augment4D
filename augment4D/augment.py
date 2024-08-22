# Author: Joydeep Munshi
# Augmentation utility funciton for crystal4D
"""
This is augmentation pipeline for 4DSTEM/STEM diffraction (CBED) images.
The different available augmentation includes elliptic distrotion, plasmonic background
noise and poisson shot noise. The elliptic distortion is recommeded to be applied on
CBED, Probe, Potential Vg and Qz tilts while other noise are only applied on CBED input.
"""
import os
import secrets
import time

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from augment4D.interpolate import dense_image_warp


class Image_Augmentation(object):
    def __init__(
        self,
        add_bkg: bool = False,
        bkg_weight: list[float] | float = [0.01, 0.1],
        bkg_q: list[float] | float = [0.01, 0.1],
        add_shot: bool = False,
        e_dose: list[float] | float = [1e5, 1e10],
        add_shift: bool = False,
        xshift: list[float] | float = [0, 10],
        yshift: list[float] | float = [0, 10],
        add_ellipticity: bool = False,
        ellipticity_scale: list[float] | float = [0, 0.15],
        add_salt_and_pepper: bool = False,
        salt_and_pepper: list[float] | float = [0, 1e-3],
        verbose: bool = False,
        log_file: os.PathLike = "./logs/augment_log.csv",
        device: str = "cpu",
        seed: int | None = None,
        magnify_only: bool = True,
    ):
        """Init augmentation, can set params here or with .set_params()

        Args:
            add_bkg (bool, optional): Whether or not to add inelastic plasmon backgorund. Defaults to False.
            bkg_weight (float, optional): Range of values for weighting the plasmon background. Defaults to 0.1.
            bkg_q (float, optional): Range of characteristic frequency of plasmon background. Defaults to 0.1.
            add_shot (bool, optional): Whether or not to add Poisson noise. Defaults to False.
            e_dose (int, optional): Range of total dose for shot noise, total dose in units of e. Defaults to 1000.
            add_shift (bool, optional): Whether or not to apply sub-pixel shifting. Defaults to False.
            xshift_range (int, optional): Range of absolute x shift in pixels. Defaults to 1.
            yshift_range (int, optional): Range of absolute y shift in pixels. Defaults to 1.
            add_ellipticity (bool, optional): Whether or not to add elliptic transforms. Defaults to False.
            ellipticity_scale (float, optional): Range of weight of elliptic transforms. Defaults to 0.1.
            add_salt_and_pepper (_type_, optional): Whether or not to add salt & pepper noise. Defaults to None.
            salt_and_pepper (_type_, optional): Fraction of image to noise. Defaults to 1e-3.
            verbose (bool, optional): Whether to include print statements. Defaults to False.
            log_file (os.PathLike, optional): Log file to print to. Defaults to "./logs/augment_log.csv".
            device (str, optional): gpu or cpu. Defaults to "cpu".
            seed (int | None, optional): random seed used by RNG. Defaults to None.
            magnify_only (bool, optional): If True forces exx and eyy <= 1, this prevents artifacts
                that arise from output images smaller than input. Defaults to True.
        """

        self.device = device.lower()
        if self.device == "gpu":
            self._xp = cp
        else:
            self._xp = np

        if seed is not None:
            self._rng_seed = seed
        else:
            self._rng_seed = secrets.randbits(128)
        self._rng = np.random.default_rng(self._rng_seed)
        self._magnify_only = magnify_only

        self.set_params(
            add_bkg,
            bkg_weight,
            bkg_q,
            add_shot,
            e_dose,
            add_shift,
            xshift,
            yshift,
            add_ellipticity,
            ellipticity_scale,
            add_salt_and_pepper,
            salt_and_pepper,
        )
        self.generate_params()

        self.verbose = verbose
        self.log_file = log_file

        with open(self.log_file, "a") as f:
            f.write(
                f"bkg_weight,bkg_q,e_dose,xshift,yshift,exx,eyy,exy,salt_and_pepper,rng_seed\n"
            )

    def set_params(
        self,
        add_bkg: bool = False,
        bkg_weight: list[float] | float = [0.01, 0.1],
        bkg_q: list[float] | float = [0.01, 0.1],
        add_shot: bool = False,
        e_dose: list[float] | float = [1e5, 1e10],
        add_shift: bool = False,
        xshift: list[float] | float = [0, 10],
        yshift: list[float] | float = [0, 10],
        add_ellipticity: bool = False,
        ellipticity_scale: list[float] | float = [0, 0.15],
        add_salt_and_pepper: bool = False,
        salt_and_pepper: list[float] | float = [0, 1e-3],
    ):
        """Set which augmentations to include and what the ranges of weight values will be used by
           each noise or augmentation type.

        Args:
            add_bkg (bool, optional): Whether or not to add inelastic plasmon backgorund. Defaults to False.
            bkg_weight (float, optional): Range of values for weighting the plasmon background. Defaults to 0.1.
            bkg_q (float, optional): Range of characteristic frequency of plasmon background. Defaults to 0.1.
            add_shot (bool, optional): Whether or not to add Poisson noise. Defaults to False.
            e_dose (int, optional): Range of total dose for shot noise, total dose in units of e. Defaults to 1000.
            add_shift (bool, optional): Whether or not to apply sub-pixel shifting. Defaults to False.
            xshift_range (int, optional): Range of absolute x shift in pixels. Defaults to 1.
            yshift_range (int, optional): Range of absolute y shift in pixels. Defaults to 1.
            add_ellipticity (bool, optional): Whether or not to add elliptic transforms. Defaults to False.
            ellipticity_scale (float, optional): Range of weight of elliptic transforms. Defaults to 0.1.
            add_salt_and_pepper (_type_, optional): Whether or not to add salt & pepper noise. Defaults to None.
            salt_and_pepper (_type_, optional): Fraction of image to noise. Defaults to 1e-3.
        """
        xp = self._xp

        self.add_bkg = add_bkg
        if self.add_bkg:
            self._bkg_weight_range = self._check_input(bkg_weight)
            self._bkg_q_range = self._check_input(bkg_q)
        else:
            self._bkg_weight_range = [0, 0]
            self._bkg_q_range = [0, 0]
            self.bkg_weight = 0
            self.bkg_q = 0

        self.add_shot = add_shot
        if self.add_shot:
            self._e_dose_range = self._check_input(e_dose)
        else:
            self._e_dose_range = [xp.inf, xp.inf]
            self.e_dose = xp.inf

        self.add_pattern_shift = add_shift
        if self.add_pattern_shift:
            self._xshift_range = self._check_input(xshift)
            self._yshift_range = self._check_input(yshift)
        else:
            self._xshift_range = [0, 0]
            self._yshift_range = [0, 0]
            self.xshift = 0
            self.yshift = 0

        self.add_ellipticity = add_ellipticity
        if self.add_ellipticity:
            self._ellipticity_scale_range = self._check_input(ellipticity_scale)
        else:
            self._ellipticity_scale_range = [0, 0]
            self.exx = 1
            self.eyy = 1
            self.exy = 0

        self.add_salt_and_pepper = add_salt_and_pepper
        if add_salt_and_pepper:
            self._salt_and_pepper_range = self._check_input(salt_and_pepper)
        else:
            self._salt_and_pepper_range = [0, 0]
            self.salt_and_pepper = 0

    def generate_params(self):
        """
        Assign new random values for each parameter based on the ranges specified with set_params
        """

        if self.add_bkg:
            self.bkg_weight = self._rand_from_range(self._bkg_weight_range)
            self.bkg_q = self._rand_from_range(self._bkg_q_range)
        else:
            self.bkg_weight = 0
            self.bkg_q = 0

        if self.add_shot:
            self.e_dose = self._rand_from_range(self._e_dose_range)
        else:
            self.e_dose = self._xp.inf

        if self.add_pattern_shift:
            self.xshift = self._rand_from_range(self._xshift_range, negative=True)
            self.yshift = self._rand_from_range(self._xshift_range, negative=True)
        else:
            self.xshift = 0
            self.yshift = 0

        if self.add_ellipticity:
            self.ellipticity_scale = self._rand_from_range(self._ellipticity_scale_range, negative=False) 
            self.exx = self._rng.normal(loc=1, scale=self.ellipticity_scale)
            self.eyy = self._rng.normal(loc=1, scale=self.ellipticity_scale)
            self.exy = self._rng.normal(loc=0, scale=self.ellipticity_scale)
            if self._magnify_only:
                self.exx = min(self.exx, 2 - self.exx)
                self.eyy = min(self.eyy, 2 - self.eyy)
        else:
            self.ellipticity_scale = 0
            self.exx = 1
            self.eyy = 1
            self.exy = 0

        if self.add_salt_and_pepper:
            self.salt_and_pepper = self._rand_from_range(self._salt_and_pepper_range)
        else:
            self.salt_and_pepper = 0
        return

    def print_params(self, print_all=False):
        print("Augmentation summary:\n", end="\r")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n", end="\r")
        print(f"Adding background: {self.add_bkg}")
        if self.add_bkg or print_all:
            print(
                f"\tWeighted background range: {self._bkg_weight_range} "
                + f"| current value: {self.bkg_weight:.3f}"
            )
            print(
                f"\tBackground plasmon range: {self._bkg_q_range} | current value: {self.bkg_q:.3f}"
            )
        print(f"Adding shot noise: {self.add_shot}")
        if self.add_shot or print_all:
            print(f"\te Dose range: [{self._e_dose_range[0]:.1e}, {self._e_dose_range[1]:.1e}] | current value {self.e_dose:.2e}")
        print(f"Adding shift: {self.add_pattern_shift}")
        if self.add_pattern_shift or print_all:
            print(f"\tPattern shift range (x,y): {self._xshift_range}, {self._yshift_range}")
            print(f"\tPattern shift current values (x,y): {self.xshift:.2f}, {self.yshift:.2f}")
        print(f"Adding elliptic scaling: {self.add_ellipticity}")
        if self.add_ellipticity or print_all:
            print(
                f"\tEllipticity scaling range: {self._ellipticity_scale_range} "
                + f"| current value: {self.ellipticity_scale:.2f}"
            )
            print(
                f"\tEllipticity params (exx, eyy, exy): ({self.exx:.2f}, {self.eyy:.2f}, "
                + f"{self.exy:.2f})"
            )
        print(f"Adding salt & pepper: {self.add_salt_and_pepper}")
        if self.add_salt_and_pepper or print_all:
            print(
                f"\tSalt & pepper range: [{self._salt_and_pepper_range[0]:.1e}, {self._salt_and_pepper_range[1]:.1e}]  "
                + f"| current value: {self.salt_and_pepper:.2e}"
            )
        print(f"Random seed: {self._rng_seed}")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n", end="\r")

    def augment_img(self, inputs, probe=None):
        """
        Apply augmentations to a stack of images.
        """
        start_time = time.time()
        input_shape = inputs.shape
        noised = inputs
        if self.add_bkg:
            noised = self._apply_bkg(inputs, probe)

        if self.add_shot:
            noised = self._apply_shot(noised)

        if self.add_ellipticity or self.add_pattern_shift:
            noised = self._apply_elastic(noised)

        if self.add_salt_and_pepper:
            noised = self._apply_salt_and_pepper(noised)

        t = time.time() - start_time

        if self.verbose:
            self.print_params()
            print(
                f"Augmentation Status: it took {t/60:.1e} minutes to augment "
                + f"{input_shape[0]} images."
            )
        self.write_logs()

        return noised

    def _apply_shot(self, inputs):
        """
        Apply Shot noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        offset = image.min()
        image -= offset
        image /= image.sum()
        return self._rng.poisson(image * self.e_dose) + offset

    def _apply_elastic(self, inputs):
        """
        Elliptic distortion and bilinear interpolated x/y shifts
        """
        image = self._as_array(inputs)
        batch_size, height, width, _channels = image.shape
        xp = self._xp

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

        imageOut = dense_image_warp(image, batched_flow_grid, xp=self._xp)

        return imageOut

    def _apply_bkg(self, inputs, probe=None):
        """
        Apply background plasmon noise
        """
        xp = self._xp
        image = xp.asarray(inputs)
        probe = xp.asarray(probe) if probe is not None else None
        batch_size, height, width, channels = image.shape

        qx, qy = self._get_qx_qy([height, width])
        CBEDbg = 1.0 / (qx**2 + qy**2 + self.bkg_q**2)
        CBEDbg = xp.squeeze(CBEDbg)
        CBEDbg = CBEDbg / xp.sum(CBEDbg)

        CBEDbg = xp.expand_dims(CBEDbg, axis=0)
        CBEDbg = xp.repeat(CBEDbg, batch_size, axis=0)
        CBEDbg = xp.expand_dims(CBEDbg, axis=-1)
        CBEDbg = xp.repeat(CBEDbg, channels, axis=0)

        CBEDbg = xp.transpose(CBEDbg, (3, 0, 1, 2))
        probe = xp.transpose(probe, (3, 0, 1, 2))

        CBEDbg_ff = xp.fft.fft2(CBEDbg).astype(xp.complex64)
        probe_ff = xp.fft.fft2(probe).astype(xp.complex64)
        mul_ff = CBEDbg_ff * probe_ff

        CBEDbgConv = xp.fft.fftshift(xp.fft.ifft2(mul_ff), axes=[2, 3])
        CBEDbgConv = xp.transpose(CBEDbgConv, (1, 2, 3, 0))

        CBEDout = (
            image.astype(xp.float32) * (1 - self.bkg_weight) + CBEDbgConv.real * self.bkg_weight
        )

        return CBEDout

    def _fourier_shift_ar(self, inputs):
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

    def _apply_salt_and_pepper(self, inputs):
        offset = inputs.min()
        ptp = (inputs - offset).max()
        im_sp = self._get_salt_and_pepper(inputs, self.salt_and_pepper)            
        im_sp = (im_sp * ptp) + offset
        return im_sp

    def _get_salt_and_pepper(self, image, amount=1e-3, salt_vs_pepper=1.0, pepper_val=0, salt_val=None):
        """
        expects normalized input [0,1]
        based off skimage implementation
        """
        out = image.copy()
        if salt_val is None:
            salt_val = image.max() 
        
        flipped = self._rng.random(out.shape) <= amount
        salted = self._rng.random(out.shape) <= salt_vs_pepper
        peppered = ~salted
        out[flipped & salted] = salt_val
        out[flipped & peppered] = pepper_val
        return out

    def write_logs(self):
        with open(self.log_file, "a") as f:
            f.write(
                f"{self.bkg_weight},{self.bkg_q},{self.e_dose},{self.xshift},{self.yshift},"
                + f"{self.exx},{self.eyy},{self.exy},{self.salt_and_pepper},{self._rng_seed}\n"
            )

    @staticmethod
    def _check_input(inp):
        if hasattr(inp, "__len__"):
            assert len(inp) == 2 and inp[0] <= inp[1], f"Bad value range give {inp}"
            out = inp
        else:
            out = [inp, inp]
        return out

    def _rand_from_range(self, val_range, negative=False):
        val = self._rng.random() * (val_range[1] - val_range[0]) + val_range[0]
        if negative:
            val *= self._rng.choice([1, -1])
        return val

    def _as_array(self, ar):
        return self._xp.asarray(ar)

    def _get_qx_qy(self, input_shape, pixel_size_AA=0.20):
        """
        get qx, qy
        """
        xp = self._xp
        N = input_shape
        qx = xp.sort(xp.fft.fftfreq(N[0], pixel_size_AA)).reshape((N[0], 1, 1))
        qy = xp.sort(xp.fft.fftfreq(N[1], pixel_size_AA)).reshape((1, N[1], 1))
        return qx, qy

    def _make_fourier_coord(self, Nx, Ny, pixelSize):
        """
        Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
        """
        xp = self._xp
        if hasattr(pixelSize, "__len__"):
            assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
            pixelSize_x = pixelSize[0]
            pixelSize_y = pixelSize[1]
        else:
            pixelSize_x = pixelSize
            pixelSize_y = pixelSize

        qx = xp.fft.fftfreq(Nx, pixelSize_x)
        qy = xp.fft.fftfreq(Ny, pixelSize_y)
        qy, qx = xp.meshgrid(qy, qx)
        return qx, qy
