import numpy as np
import PIL

from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin


class spnoise_job(JobArgs):
    def __init__(self, coverage=0.011, amplitude=0.125, **kwargs):
        super().__init__(**kwargs)
        self.coverage = coverage
        self.amplitude = amplitude


class SPNoisePlugin(Plugin):
    def title(self):
        return "My Plugin"

    def describe(self):
        return "A plugin to inject salt & pepper noise."

    def init(self):
        pass

    def sp_noise(self, j: spnoise_job):
        """
        Add salt and pepper noise to image
        """
        pil = j.input.image

        img = np.array(pil)
        # if mask is not None:
        #     mask = np.array(mask.convert("RGB"))

        original_dtype = img.dtype

        # Derive the number of intensity levels from the array datatype.
        intensity_levels = 2 ** (img[0, 0].nbytes * 8)
        min_intensity = 0
        max_intensity = intensity_levels - 1

        random_image_arr = np.random.choice(
                [min_intensity, 1, np.nan],
                p=[j.coverage / 2, 1 - j.coverage, j.coverage / 2],
                size=img.shape
        )

        # This results in an image array with the following properties:
        # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
        # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
        # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
        # `arr.astype(np.float)` to make sure np.nan is a valid value.
        salt_and_peppered_arr = img.astype(float) * random_image_arr

        # Since we want SALT instead of NaN, we replace it.
        # We cast the array back to its original dtype so we can pass it to PIL.
        salt_and_peppered_arr = np.nan_to_num(
                salt_and_peppered_arr,
                nan=max_intensity
        ).astype(original_dtype)

        ret = salt_and_peppered_arr

        # if mask is not None:
        #     zeros = np.zeros_like(img)
        # ret = ret * mask + zeros * (1-mask)

        pil = PIL.Image.fromarray(ret)

        return pil
