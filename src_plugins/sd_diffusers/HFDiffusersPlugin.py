import torch

from classes.convert import load_pil, pil2cv
from lib.devices import device
from src_core.plugins import plugjob
from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin

from diffusers import ControlNetModel, LMSDiscreteScheduler, StableDiffusionControlNetPipeline, StableDiffusionImageVariationPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, VersatileDiffusionPipeline

class HFDiffusersPlugin(Plugin):
    def title(self):
        return "HF Diffusers"

    def load(self):
        print("Loading HF Diffusers...")
        # self.ptxt = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
        #                                                     safety_checker=None,
        #                                                     requires_safety_checker=False,
        #                                                     torch_dtype=torch.float32)

        # self.pvar = StableDiffusionImageVariationPipeline.from_pretrained(
        #         "lambdalabs/sd-image-variations-diffusers",
        #         revision="v2.0",
        #         safety_checker=None,
        #         requires_safety_checker=False)
        # self.pvar.enable_model_cpu_offload()
        # self.pvar.enable_attention_slicing()

        # self.ptxt = StableDiffusionControlNetPipeline.from_pretrained(
        #         "runwayml/stable-diffusion-v1-5",
        #         safety_checker=None,
        #         requires_safety_checker=False,
        #         controlnet=[
        #             ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
        #             # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16),
        #             ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
        #         ],
        #         torch_dtype=torch.float16).to("cuda")

        # self.ptxt.enable_model_cpu_offload()
        # self.ptxt.enable_attention_slicing(1)
        # pipe.enable_vae_tiling()
        # pipe.enable_sequential_cpu_offload()
        # pipe.unet.to(memory_format=torch.channels_last)  # in-place operation

        # self.pvar.scheduler = UniPCMultistepScheduler.from_config(self.pvar.scheduler.config)
        # self.pvar.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)

        # self.pimg = StableDiffusionImg2ImgPipeline(
        #         vae=self.ptxt.vae,
        #         text_encoder=self.ptxt.text_encoder,
        #         tokenizer=self.ptxt.tokenizer,
        #         unet=self.ptxt.unet,
        #         scheduler=self.ptxt.scheduler,
        #         safety_checker=None,
        #         feature_extractor=None,
        #         requires_safety_checker=False)
        self.pvd = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion",
                                                               safety_checker=None,
                                                               requires_safety_checker=False)
        self.pvd.scheduler = UniPCMultistepScheduler.from_config(self.pvd.scheduler.config)
        self.pvd.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
        self.pvd.enable_model_cpu_offload()
        self.pvd.enable_attention_slicing(1)

        print("Diffusers pipe loaded!")

    def txt2img(self,
                prompt: str = None,
                negprompt: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                **kwargs):
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
        from rendering import renderer
        session = renderer.session

        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = self.ptxt(prompt=prompt,
                          image=image or session.image,
                          guidance_scale=cfg,
                          num_inference_steps=steps,
                          generator=generator).images[0]

        return pil2cv(image)

    def img2img(self,
                prompt: str = None,
                negprompt: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                **kwargs):
        # return self.txt2img(self, job)
        from rendering import renderer
        session = renderer.session

        generator = torch.Generator(device="cpu").manual_seed(seed)
        images = self.pimg(prompt=prompt,
                           image=image or session.image,
                           strength=chg,
                           guidance_scale=cfg,
                           num_inference_steps=steps,
                           generator=generator,
                           controlnet_conditioning_scale=ccg
                           ).images

        return pil2cv(images[0])

    def imvar(self,
              cfg: float = 7.5,
              image: str = None,
              ccg: float = 1.0,
              chg: float = 0.5,
              steps: int = 30,
              seed: int = 0,
              **kwargs):
        from torchvision.transforms import transforms

        if image is not None:
            image = load_pil(image)
        if image is None:
            image = self.session.image

        # tform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(
        #             (224, 224),
        #             interpolation=transforms.InterpolationMode.BICUBIC,
        #             antialias=False,
        #     ),
        #     transforms.Normalize(
        #             [0.48145466, 0.4578275, 0.40821073],
        #             [0.26862954, 0.26130258, 0.27577711]),
        # ])
        # inp = tform(image).to(device).unsqueeze(0)
        #
        # out = self.pvar(inp, guidance_scale=cfg, num_inference_steps=steps)
        # ret = out["images"][0]

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pvd.image_variation(image,
                                         width=image.width,
                                         height=image.height,
                                         negative_prompt=None,
                                         guidance_scale=cfg,
                                         num_inference_steps=steps,
                                         generator=generator)
        ret = result["images"][0]

        return pil2cv(ret)
