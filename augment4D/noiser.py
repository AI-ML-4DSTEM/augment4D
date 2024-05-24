import numpy as np
import skimage
from skimage import exposure
from skimage.restoration import estimate_sigma
from scipy import ndimage as ndi
from scipy import stats
from skimage.util import random_noise as skrandom_noise
import torch
from pathlib import Path


class image_noiser:
    """
    Applies sequence of image noising steps to an image(s)

    gauss : percent noise value, float, ~20
    poisson: [0,1] float, ~0.3-0.5
    salt_and_pepper: ~50, float
    blur: sigma value (pixels), float, ~5
    jitter : [0, ...3?] float
    contrast: gamma level, [0,1] for brighter (doesnt do much), and > 1 for darker. ~2
    bkg float, [0-?]

    """

    def __init__(
        self,
        gauss=0,  # percent noise to add
        poisson=0,
        salt_and_pepper=0,
        blur=0,
        jitter=0,
        contrast=1,  # 1 is no change
        bkg=0,
        seed=None,
    ) -> None:
        """
        initialize image noising parameters
        """
        self.gauss = gauss
        self.poisson = poisson
        self.salt_and_pepper = salt_and_pepper
        self.blur = blur
        self.jitter = jitter
        self.contrast = contrast
        self.bkg = bkg
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def run(self, image):
        if isinstance(image, torch.Tensor):
            inp_torch = True
            inp_shape = image.shape
            device = image.device
            image = image.cpu().detach().numpy()
        else:
            inp_torch = False
            inp_shape = image.shape

        image = image.squeeze()
        if np.ndim(image) != 2:
            raise ValueError(
                f"Expected array of dimension 2, got image of shape {image.shape}"
            )
        self.h, self.w = image.shape

        if self.gauss != 0:
            image = self.apply_gauss(image)
        if self.poisson != 0:
            image = self.apply_poisson(image)
        if self.salt_and_pepper != 0:
            image = self.apply_salt_and_pepper(image)
        if self.blur != 0:
            image = self.apply_blur(image)
        if self.jitter != 0:
            image = self.apply_jitter(image)
        if self.contrast != 1 and self.contrast is not None:
            image = self.apply_contrast(image)
        if self.bkg != 0:
            image = self.apply_bkg(image)

        if inp_torch:
            image = torch.tensor(image, device=device)

        image = image.reshape(inp_shape)
        return image

    def apply_gauss(self, image):
        sigma = self.gauss / 200 * np.mean(image)
        noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
        return image + noise

    def apply_poisson(self, image):
        offset = np.min(image)
        omax = np.max(image - image.min())
        im = (image - offset) / omax  # norm image
        vals = len(np.unique(im))
        vals = (self.poisson ** (-1 / 3)) ** np.ceil(np.log2(vals))
        noisy = np.random.poisson((im) * vals) / float(vals)
        return (noisy * omax) + offset

    def apply_salt_and_pepper(self, image):
        minval = image.min()
        imfac = (image - minval).max()
        im_sp = skrandom_noise(
            (image - minval) / imfac,
            mode="s&p",
            amount=self.salt_and_pepper * 1e-3,
            seed=self.seed,
        )
        im_sp = (im_sp * imfac) + minval
        return im_sp

    def apply_blur(self, image):
        return ndi.gaussian_filter(image, self.blur, mode="wrap")

    def apply_jitter(self, image):
        shift_arr = stats.poisson.rvs(self.jitter, loc=0, size=self.h)
        im_jitter = np.array([np.roll(row, z) for row, z in zip(image, shift_arr)])
        return im_jitter

    def apply_contrast(self, image):
        if self.contrast == 0:
            raise ValueError("Contrast should not be 0.")
        return exposure.adjust_gamma(image, self.contrast)

    def apply_bkg(self, image):
        def gauss2d(xy, x0, y0, a, b, fwhm):
            return np.exp(
                -np.log(2) * (a * (xy[0] - x0) ** 2 + b * (xy[1] - y0) ** 2) / fwhm**2
            )

        h, w = image.shape
        x, y = np.meshgrid(np.linspace(0, h, h), np.linspace(0, w, w), indexing="ij")
        x0 = np.random.randint(0, h - h // 4)
        y0 = np.random.randint(0, w - w // 4)
        a, b = np.random.randint(10, 20, 2) / 10
        fwhm = np.random.randint(min([h, w]) // 4, min([h, w]) - min([h, w]) // 2)
        Z = gauss2d([x, y], x0, y0, a, b, fwhm)
        fac = 0.05 * (np.random.randint(0, 2) * 2 - 1)
        bkg = self.bkg * fac * np.random.randint(-10, 10) * Z
        return image + bkg

    def log(self, fpath):
        fpath = Path(fpath).resolve()
        if fpath.suffix not in [".txt", ".log"]:
            fpath = fpath.parent / (fpath.name + ".txt")
        with open(fpath, "w") as f:
            f.write(f"gauss: {self.gauss}\n")
            f.write(f"jitter: {self.jitter}\n")
            f.write(f"poisson: {self.poisson}\n")
            f.write(f"salt_and_pepper: {self.salt_and_pepper}\n")
            f.write(f"blur: {self.blur}\n")
            f.write(f"contrast: {self.contrast}\n")
            f.write(f"bkg: {self.bkg}\n")
            f.write(f"seed: {self.seed}\n")
        return fpath


def get_percent_noise(noisy, truth=None):
    """
    for 2 or 3 dimension input, if 3D assumes channel axis is axis 0
    """
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.cpu().detach().numpy()
        if isinstance(truth, torch.Tensor):
            truth = truth.cpu().detach().numpy()

    if truth is None:
        if noisy.ndim == 3:
            return np.mean(
                estimate_sigma(noisy, channel_axis=0)
                / (np.mean(noisy, axis=(-2, -1)))
                * 200
            )
        elif noisy.ndim == 2:
            return estimate_sigma(noisy) / (np.mean(noisy)) * 200
        else:
            raise NotImplementedError
    else:
        assert noisy.shape == truth.shape
        return np.mean(
            np.std(noisy - truth, axis=(-2, -1)) / np.mean(truth, axis=(-2, -1)) * 200
        )