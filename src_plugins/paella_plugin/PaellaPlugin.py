import torch

from src_core import installing
from src_core.classes import paths
from src_core.classes.Plugin import Plugin
from src_core.classes.prompt_job import prompt_job
from src_core.plugins import plugjob
from src_plugins.paella_plugin import colab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class paella_job(prompt_job):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Paella"

class PaellaPlugin(Plugin):
    def title(self):
        return "paella"

    def describe(self):
        return "Paella is a neural network that can generate images from text descriptions."

    def init(self):
        pass

    def load(self):
        pt = self.res("paella.pt")
        if not pt.exists():
            pt = paths.get_first_match(self.res(), suffix='.pt')
        colab.load(pt)

    def unload(self):
        pass

    @plugjob
    def txt2img(self, j:paella_job):
        return colab.txt2img(j.prompt)


