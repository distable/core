import numpy as np
import PIL
from PIL import ImageEnhance, ImageStat

from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin
from src_core.plugins import plugjob

# @markdown Automatic Brightness Adjustment ------------------------------------------------------------
# @markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
# @markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
# @markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
# @markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds

# Credit: https://github.com/lowfuel/progrockdiffusion

class progrock_brightness_job(JobArgs):
    def __init__(self,
                 enable_adjust_brightness=False,
                 high_brightness_threshold=180,
                 high_brightness_adjust_ratio=0.97,
                 high_brightness_adjust_fix_amount=2,
                 max_brightness_threshold=254,
                 low_brightness_threshold=40,
                 low_brightness_adjust_ratio=1.03,
                 low_brightness_adjust_fix_amount=2,
                 min_brightness_threshold=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.enable_adjust_brightness = enable_adjust_brightness
        self.high_brightness_threshold = high_brightness_threshold
        self.high_brightness_adjust_ratio = high_brightness_adjust_ratio
        self.high_brightness_adjust_fix_amount = high_brightness_adjust_fix_amount
        self.max_brightness_threshold = max_brightness_threshold
        self.low_brightness_threshold = low_brightness_threshold
        self.low_brightness_adjust_ratio = low_brightness_adjust_ratio
        self.low_brightness_adjust_fix_amount = low_brightness_adjust_fix_amount
        self.min_brightness_threshold = min_brightness_threshold


def get_stats(image):
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / len(stat.mean)
    contrast = sum(stat.stddev) / len(stat.stddev)
    return brightness, contrast


class ProgrockbrightnessPlugin(Plugin):
    def title(self):
        return "progrockbrightness"

    def describe(self):
        return ""

    @plugjob
    def progrock_brightness(image, j:progrock_brightness_job):
        brightness, contrast = get_stats(image)
        if brightness > j.high_brightness_threshold:
            filter = ImageEnhance.Brightness(image)
            image = filter.enhance(j.high_brightness_adjust_ratio)
            image = np.array(image)
            image = np.where(image > j.high_brightness_threshold, image - j.high_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
            image = PIL.Image.fromarray(image)
        if brightness < j.low_brightness_threshold:
            filter = ImageEnhance.Brightness(image)
            image = filter.enhance(j.low_brightness_adjust_ratio)
            image = np.array(image)
            image = np.where(image < j.low_brightness_threshold, image + j.low_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
            image = PIL.Image.fromarray(image)

        image = np.array(image)
        image = np.where(image > j.max_brightness_threshold, image - j.high_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
        image = np.where(image < j.min_brightness_threshold, image + j.low_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
        image = PIL.Image.fromarray(image)
        return image
