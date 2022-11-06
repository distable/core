import PIL

from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin
from src_core.plugins import plugjob


class edgedet_job(JobArgs):
    pass

class EdgePlugin(Plugin):
    def title(self):
        return "Edge Detection"

    def describe(self):
        return "A plugin for edge detection."

    def init(self):
        pass

    @plugjob
    def edge_detection(self, j:edgedet_job):
        from PIL import Image
        import numpy as np
        import scipy.signal as sg

        img = np.asarray(j.input.image)
        img = img[:, :, 0]
        img = img.astype(np.int16)

        edge = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])

        results = sg.convolve(img, edge, mode='same')
        results[results > 127] = 255
        results[results < 0] = 0

        results = results.astype(np.uint8)
        comp = np.mean(results)
        return PIL.Image.fromarray(results), comp