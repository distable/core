import os

import cv2
import numpy as np
# import wand.image
from PIL import Image, ImageEnhance

from src_core.classes import paths
from src_core.installing import run
from src_core.classes.JobArgs import JobArgs
from src_core.classes.convert import cv2pil, pil2cv
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin
from src_plugins.disco_party.maths import clamp


class hsvc_job(JobArgs):
    def __init__(self, hue=None, sat=None, val=None, contrast=None, **kwargs):
        super().__init__(**kwargs)
        self.hue: int = hue
        self.sat: int = sat
        self.val: int = val
        self.contrast: int = contrast


class hsvc_match_job(hsvc_job):
    def __init__(self, speed=0.5, **kwargs):
        super().__init__(**kwargs)
        self.speed = speed


class hsvc_add_job(hsvc_job):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class denoise_job(JobArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class contrast_job(JobArgs):
    def __init__(self, contrast=0, **kwargs):
        super().__init__(**kwargs)
        self.contrast = contrast


class MagickPlugin(Plugin):
    def title(self):
        return "Magick Plugin"

    def describe(self):
        return "Run some color corrections with image magick"

    def init(self):
        pass

    def install(self):
        # TODO this functionality should be in installing.py (with proper user input and stuff)
        # check if pacman is installed
        if run("pacman") == 0:
            # Install imagemagick if not installed
            if run("pacman -Q imagemagick") != 0:
                run("pacman -S imagemagick")

        # try with apt-get
        elif run("apt-get") == 0:
            # Install imagemagick if not installed
            if run("apt-get -Q imagemagick") != 0:
                run("apt-get install imagemagick")

        # oh well
        else:
            print("No package manager found to install ImageMagick")
            return

        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def get_avg_hsv(self, pil) -> (float, float, float):
        img_hsv = cv2.cvtColor(pil2cv(pil), cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0].mean() / 255
        sat = img_hsv[:, :, 1].mean() / 255
        val = img_hsv[:, :, 2].mean() / 255
        return hue, sat, val

    def magick(self, pil, command="+sigmoidal-contrast 5x-3%"):
        src = (paths.root / "_magick.png").as_posix()
        dst = (paths.root / "_out.png").as_posix()

        pil.save(src)
        os.system(f"convert '{src}' {command} '{dst}'")
        ret = Image.open('_out.png').convert('RGB')
        os.remove(src)
        os.remove(dst)

        return ret

    def __call__(self, pil, *args, **kwargs):
        return self.magick(pil, *args, **kwargs)

    @plugjob
    def dmagick(self, args: hsvc_job):
        pil = args.image
        if pil is None:
            return None

        # img = magick(img, '-auto-level')

        pil = self.magick(pil, '-contrast-stretch')
        pil = self.magick(pil, '-auto-level')
        # pil = magick(pil, '-brightness-contrast 0x4%')
        pil = self.magick(pil, f'-modulate 100,{args.sat},{args.hue}')
        # pil = magick(img, '-normalize')
        return pil

    @plugjob
    def cc(self, j: hsvc_match_job):
        """
        Keep the hue, brightness or saturation around a certain value.
        Input targets are expected to be normalized 0-1
        """
        img = j.session.image
        if img is None:
            return None

        img_hsv = cv2.cvtColor(pil2cv(img), cv2.COLOR_BGR2HSV)
        hue_mean = img_hsv[:, :, 0].mean() / 255
        sat_mean = img_hsv[:, :, 1].mean() / 255
        val_mean = img_hsv[:, :, 2].mean() / 255
        contrast_mean = img_hsv[:, :, 2].std() / 255

        j.hue = j.hue if j.hue is not None else hue_mean
        j.sat = j.sat if j.sat is not None else sat_mean
        j.val = j.val if j.val is not None else val_mean
        j.contrast = j.contrast if j.contrast is not None else contrast_mean

        hue = (j.hue - hue_mean) * j.speed
        val = (j.val - val_mean) * j.speed
        sat = (j.sat - sat_mean) * j.speed
        contrast = (j.contrast - contrast_mean) * j.speed

        hue = 255 + hue * 255
        sat = 255 + sat * 255
        val = 255 + val * 255

        # Equivalent with wand
        self.add_hsv(hue, sat, val)

    @plugjob
    def ccadd(self, j: hsvc_add_job):
        if j.session.image is None:
            return None

        # Add hue, saturation and value (all normalized in 0 to 1)
        # Use ImageEnhance for image manipulation
        img = j.session.image
        # img = ImageEnhance.Color(img).enhance(j.sat)
        # img = ImageEnhance.Brightness(img).enhance(j.val)
        # img = ImageEnhance.Contrast(img).enhance(j.contrast)

        # Equivalent with numpy
        img = self.add_hsv(j.hue, img, j.sat, j.val)

        return cv2pil(img)

    def add_hsv(self, hue, img, sat, val):
        img_hsv = cv2.cvtColor(pil2cv(img), cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 0] += int(hue % 255)
        img_hsv[:, :, 1] += int(clamp(sat, -255, 255))
        img_hsv[:, :, 2] += int(clamp(val, -255, 255))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img

    @plugjob
    def denoise(self, j: denoise_job):
        if j.session.image is None:
            return None

        return self.magick(j.session.image, f'-enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance')


    @plugjob
    def contrast(self, j: contrast_job):
        if j.session.image is None:
            return None

        return ImageEnhance.Contrast(j.session.image).enhance(j.contrast)

    # @plugjob
    # def distort(self, j: distort_job):
    #     if j.session.image is None:
    #         return None
    #
    #     with wand.image.Image.from_array(np.array(j.session.image)) as img:
    #         # Grid distortion
    #         return wnd_to_pil(img)


def pil_to_wnd(pil):
    return wand.image.Image.from_array(np.array(pil))


def wnd_to_pil(img):
    array = np.array(img)
    array = np.uint8(array * 255)
    # print(array)
    return Image.fromarray(array)


def cv2_hue(c, amount):
    amount * 255
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c


def cv2_brightness(input_img, brightness=0):
    """
        input_image:  color or grayscale image
        brightness:  -127 (all black) to +127 (all white)
            returns image of same type as input_image but with
            brightness adjusted
    """
    brightness *= 127

    img = input_img.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        cv2.convertScaleAbs(input_img, img, alpha_b, gamma_b)

    return img
