import numpy as np
import opensimplex
from PIL import Image
from skimage.util import random_noise

from src_core.classes.convert import save_png
from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin
from src_core.plugins import plugjob


class spnoise_job(JobArgs):
    def __init__(self, coverage=0.011, amplitude=0.125, **kwargs):
        super().__init__(**kwargs)
        self.coverage = coverage
        self.amplitude = amplitude


class perlin_job(JobArgs):
    def __init__(self, k=2, speed=1.0, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.speed = speed


class SPNoisePlugin(Plugin):
    def title(self):
        return "My Plugin"

    def describe(self):
        return "A plugin to inject salt & pepper noise."

    def init(self):
        pass

    @plugjob
    def perlin(self, j: perlin_job):
        src = j.session.image_cv2
        if src is None:
            return None

        # Generate perlin noise pil
        t = j.session.t * j.speed

        mask = (opensimplex.noise3array(
                np.linspace(0, 1 / j.k, j.session.width),
                np.linspace(0, 1 / j.k, j.session.height),
                np.linspace(t, t, 1),
        )[0] + 1) / 2

        noise = (opensimplex.noise3array(
                np.linspace(0, 1 / 0.001, j.session.width),
                np.linspace(0, 1 / 0.001, j.session.height),
                np.linspace(t + 30, t + 30, 1),
        )[0] + 1) / 2

        rgb = np.random.randint(0, 255, (3, j.session.height, j.session.width))


        dst = mask * noise * 255
        # Stack dst to 3 channels multiplied with random rgb
        dst = np.stack([dst*rgb[0]/255, dst*rgb[1]/255, dst*rgb[2]/255], axis=2)
        # dst = np.stack([dst, dst, dst], axis=2)

        # maskpil = PIL.Image.fromarray(np.uint8(mask * j.alpha))
        # noisepil = PIL.Image.fromarray(np.uint8(noise))

        # Create empty pil with same size as input
        # ret = PIL.Image.new('RGB', srcpil.size, (0, 0, 0))
        ret = src + (dst-127) * j.alpha
        ret = np.clip(ret, 0, 255)

        # print(maskpil, j.session.current_frame_path('perlin'))
        # save_png(Image.fromarray(np.uint8(dst)), j.session.current_frame_path('perlin'), with_async=True)

        return ret


    @plugjob
    def spnoise(self, j: spnoise_job):
        """
        Add salt and pepper noise to image
        """
        img = j.session.image_cv2
        if img is None:
            return None

        noise = random_noise(img, mode='s&p', amount=j.coverage)
        img = np.array(255 * noise, dtype='uint8')

        # # if mask is not None:
        # #     mask = np.array(mask.convert("RGB"))
        #
        # original_dtype = img.dtype
        #
        # # Derive the number of intensity levels from the array datatype.
        # intensity_levels = 2 ** (img[0, 0].nbytes * 8)
        # min_intensity = 0
        # max_intensity = intensity_levels - 1
        #
        # random_image_arr = np.random.choice(
        #         [min_intensity, 1, np.nan],
        #         p=[j.coverage / 2, 1 - j.coverage, j.coverage / 2],
        #         size=img.shape
        # )
        #
        # # This results in an image array with the following properties:
        # # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
        # # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
        # # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
        # # `arr.astype(np.float)` to make sure np.nan is a valid value.
        # salt_and_peppered_arr = img.astype(float) * random_image_arr
        #
        # # Since we want SALT instead of NaN, we replace it.
        # # We cast the array back to its original dtype so we can pass it to PIL.
        # salt_and_peppered_arr = np.nan_to_num(
        #         salt_and_peppered_arr,
        #         nan=max_intensity
        # ).astype(original_dtype)
        #
        # ret = salt_and_peppered_arr
        #
        # # if mask is not None:
        # #     zeros = np.zeros_like(img)
        # # ret = ret * mask + zeros * (1-mask)
        #
        # img = Image.fromarray(ret)

        return img
