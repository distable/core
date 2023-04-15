import torch

from src_core import installing
from src_core.classes import paths
from src_core.classes.Plugin import Plugin
from src_core.classes.prompt_job import prompt_job
from src_core.plugins import plugjob
from src_plugins.paella import colab

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
        from huggingface_hub import hf_hub_download
        colab.load(
                hf_hub_download(repo_id="dome272/Paella", filename="paella_v3.pt"),
                hf_hub_download(repo_id="dome272/Paella", filename="vqgan_f4.pt"),
                hf_hub_download(repo_id="dome272/Paella", filename="prior_v1.pt"))

    def unload(self):
        pass


    # def img2img(self, j:paella_job):
    #     return colab.img_variation(j.prompt)

    def txt2img(self, prompt='', size=None, cfg=5.0, seed=0, steps=6, **kwargs):
        import numpy as np

        if size is None:
            size = (32, 32)

        ret = colab.txt2img(prompt, steps=steps, size=size, cfg=cfg, seed=seed)
        ret = torch.cat([
            torch.cat([i for i in ret], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu()
        # ret = ret.squeeze().cpu().numpy()
        # ret = ret.transpose(1, 2, 0)
        ret = (ret * 255).numpy().astype(np.uint8)

        return ret
