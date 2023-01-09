import cv2
from PIL import Image
from skimage.exposure import match_histograms

from src_core.classes.JobArgs import JobArgs
from src_core.classes.printlib import printerr
from src_core.classes.convert import cv2pil, pil2cv
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin
from src_plugins.disco_party.maths import droplerp_np


class palette_job(JobArgs):
    def __init__(self, pal:Image.Image|str, mode='LAB', speed=1, **kwargs):
        super().__init__(**kwargs)
        self.pal = pal
        self.mode = mode
        self.speed = speed


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    elif mode == 'LAB':  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


class PalettePlugin(Plugin):
    def title(self):
        return "My Plugin"

    def describe(self):
        return "Describe me"

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    @plugjob
    def palette(self, j: palette_job):
        palpil = None
        if isinstance(j.pal, Image.Image):
            palpil = j.pal
        elif isinstance(j.pal, str):
            palpil = Image.open(j.pal)
        else:
            printerr(f"Palette must be a PIL Image or a path to an image file, got {type(j.pal)}")

        img = pil2cv(j.ctx.image)
        pal = pil2cv(palpil)
        droplerp_np(img, pal, 4, j.speed)
        retcv = maintain_colors(img, pal, j.mode)

        return cv2pil(retcv)
