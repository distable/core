import cv2
from skimage.exposure import match_histograms

from src_core.classes.JobArgs import JobArgs
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin


class palmatch_job(JobArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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
    def match(self, p: palmatch_job):
        pass # TODO we need an input pil to work with
