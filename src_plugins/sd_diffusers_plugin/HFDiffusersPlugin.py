import torch

from src_core.plugins import plugjob
from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin

from diffusers import ControlNetModel, LMSDiscreteScheduler, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler

class diffusers_job(JobArgs):
    def __init__(self,
                 negprompt: str = None,
                 cfg: float = 7.0,
                 image: str = None,
                 chg: float = 0.5,
                 steps: int = 30,
                 seed: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.negprompt = negprompt
        self.cfg = cfg
        self.image = image
        self.chg = chg
        self.steps = steps
        self.seed = seed

class diffusers_txt2img_job(diffusers_job):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class diffusers_img2img_job(diffusers_job):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HFDiffusersPlugin(Plugin):
    def title(self):
        return "HF Diffusers"


    def load(self):
        # self.ptxt = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
        #                                                     safety_checker=None,
        #                                                     requires_safety_checker=False,
        #                                                     torch_dtype=torch.float32)

        cn_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
        self.ptxt = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                requires_safety_checker=False,
                controlnet=cn_canny,
                torch_dtype=torch.float32)

        self.ptxt.enable_sequential_cpu_offload()
        # self.ptxt.enable_attention_slicing(1)

        self.ptxt.scheduler = UniPCMultistepScheduler.from_config(self.ptxt.scheduler.config)
        # self.ptxt.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)

        self.pimg = StableDiffusionImg2ImgPipeline(
                vae=self.ptxt.vae,
                text_encoder=self.ptxt.text_encoder,
                tokenizer=self.ptxt.tokenizer,
                unet=self.ptxt.unet,
                scheduler=self.ptxt.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)

        print("Diffusers pipe loaded!")

    @plugjob
    def txt2img(self, j: diffusers_txt2img_job):
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

        generator = torch.Generator(device="cpu").manual_seed(j.seed)
        image = self.ptxt(prompt=j.prompt,
                          image=j.image or j.session.image,
                          guidance_scale=j.cfg,
                          num_inference_steps=j.steps,
                          generator=generator).images[0]

        return image

    @plugjob
    def img2img(self, job: diffusers_img2img_job):
        # return self.txt2img(self, job)

        generator = torch.Generator(device="cpu").manual_seed(job.seed)
        images = self.pimg(prompt=job.prompt,
                           image=job.image or job.session.image,
                           strength=job.chg,
                           guidance_scale=job.cfg,
                           num_inference_steps=job.steps,
                           generator=generator,
                           ).images

        return images[0]

        # pipe.enable_vae_tiling()
        # pipe.enable_sequential_cpu_offload()
        # pipe.unet.to(memory_format=torch.channels_last)  # in-place operation


        # return images[0]
