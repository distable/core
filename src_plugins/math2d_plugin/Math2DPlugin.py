import os

import cv2
import numpy as np
import PIL
from PIL.Image import Image

from src_core.classes.JobParams import JobParams
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin


class zoom_job(JobParams):
    def __init__(self, zoom: float, **kwargs):
        super().__init__(**kwargs)
        self.zoom = zoom


class rot_job(JobParams):
    def __init__(self, rot: float, **kwargs):
        super().__init__(**kwargs)
        self.rot = rot


class transform2d_job(JobParams):
    def __init__(self, x: float = 0, y: float = 0, rot: float = 0, zoom: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.rot = rot
        self.zoom = zoom


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
    def transform2d(pil: Image = None, p: transform2d_job = None):
        # TODO ewwwwwwwwwww
        pil.save('tmp.png')
        img_0 = cv2.imread('tmp.png')

        center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
        trans_mat = np.float32(
                [[1, 0, p.x],
                 [0, 1, p.y]]
        )
        rot_mat = cv2.getRotationMatrix2D(center, p.rot, p.zoom)
        trans_mat = np.vstack([trans_mat, [0, 0, 1]])
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])
        transformation_matrix = np.matmul(rot_mat, trans_mat)
        img_0 = cv2.warpPerspective(
                img_0,
                transformation_matrix,
                (img_0.shape[1], img_0.shape[0]),
                borderMode=cv2.BORDER_WRAP
        )
        cv2.imwrite('tmp.png', img_0)
        pil = PIL.Image.open('tmp.png')

        # TODO ewwwwwwwwwwww
        os.remove('tmp.png')

        return pil.convert("RGB")
