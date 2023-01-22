import os

import cv2
import numpy as np
import PIL
from PIL.Image import Image

from src_core.classes.convert import cv2pil, pil2cv
from src_core.classes.JobArgs import JobArgs
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin


class zoom_job(JobArgs):
    def __init__(self, zoom: float, **kwargs):
        super().__init__(**kwargs)
        self.zoom = zoom


class rot_job(JobArgs):
    def __init__(self, rot: float, **kwargs):
        super().__init__(**kwargs)
        self.rot = rot


class transform2d_job(JobArgs):
    def __init__(self, x: float = 0, y: float = 0, rot: float = 0, zoom: float = 0, *kargs, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.rot = rot
        self.zoom = 1 + zoom


class Math2DPlugin(Plugin):
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
    def mat2d(self, p: transform2d_job = None):
        img = p.session.image_cv2
        if img is None:
            return

        center = (1 * img.shape[1] // 2, 1 * img.shape[0] // 2)
        translate = np.float32(
                [[1, 0, p.x],
                 [0, 1, p.y]]
        )
        rotate = cv2.getRotationMatrix2D(center, p.rot, p.zoom)
        transform = np.matmul(np.vstack([rotate, [0, 0, 1]]),
                              np.vstack([translate, [0, 0, 1]]))

        img = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

        return img
