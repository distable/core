import os

import cv2
from PIL import Image

from src_core.installing import run
from src_core.classes.JobArgs import JobArgs
from src_core.convert import pil2cv
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin


class hsbc_job(JobArgs):
    def __init__(self, hue=None, saturation=None, brightness=None, contrast=None, **kwargs):
        super().__init__(**kwargs)
        self.hue: int = hue
        self.saturation: int = saturation
        self.brightness: int = brightness
        self.contrast: int = contrast


class maintain_job(hsbc_job):
    def __init__(self, speed=0.5, **kwargs):
        super().__init__(**kwargs)
        self.speed = speed


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
        pil.save('_magick.png')

        os.system(f"convert _magick.png ${command} _out.png")
        os.remove('_magick.png')

        new = Image.open('_out.png')
        os.remove('_out.png')

        return new

    def __call__(self, pil, *args, **kwargs):
        return self.magick(pil, *args, **kwargs)

    @plugjob
    def dmagick(self, args: hsbc_job):
        pil = args.input.image

        # img = magick(img, '-auto-level')

        # pil = magick(pil, '-contrast-stretch')
        pil = self.magick(pil, '-auto-level')
        # pil = magick(pil, '-brightness-contrast 0x4%')
        pil = self.magick(pil, f'-modulate 100,{args.saturation},{args.hue}')
        # pil = magick(img, '-normalize')
        return pil

    @plugjob
    def cc(self, j: maintain_job):
        """
        Keep the hue, brightness or saturation around a certain value.
        Input targets are expected to be normalized 0-1
        """
        img = j.input.image

        img_hsv = cv2.cvtColor(pil2cv(img), cv2.COLOR_BGR2HSV)
        hue_mean = img_hsv[:, :, 0].mean()
        sat_mean = img_hsv[:, :, 1].mean()
        val_mean = img_hsv[:, :, 2].mean()

        j.hue = j.hue if j.hue is not None else hue_mean
        j.saturation = j.saturation if j.saturation is not None else sat_mean
        j.brightness = j.brightness if j.brightness is not None else val_mean

        hue = 100 + (j.hue*255 - hue_mean) / 255 * j.speed
        brightness = 100 + (j.brightness*255 - val_mean) / 255 * j.speed
        saturation = 100 + (j.saturation*255 - sat_mean) / 255 * j.speed

        return self.magick(img, f'-modulate {brightness},{saturation},{hue}')

    @plugjob
    def shift(self, j: hsbc_job):
        return self.magick(j.input.image, f'-modulate {j.brightness},{j.saturation},{j.hue}')
