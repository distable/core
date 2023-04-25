
import torch
from compel import Compel

from classes.convert import load_pil, pil2cv
from classes.printlib import printkw
from lib import devices
from lib.devices import device
from plugins import plugfun, plugfun_img
from src_core.rendering.hud import hud
from src_core.classes.Plugin import Plugin

from diffusers import AutoencoderKL, ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, UniPCMultistepScheduler

class HFDiffusersPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ptxt = None
        self.pimg = None
        self.pvar = None
        self.pvd = None
        self.pcontrolnet = None
        self.pcontrolnet_inpaint = None
        self.controlnets = {}

    def title(self):
        return "HF Diffusers"

    def load(self):
        print("Loading HF Diffusers ...")

        # self.pvar = StableDiffusionImageVariationPipeline.from_pretrained(
        #         "lambdalabs/sd-image-variations-diffusers",
        #         revision="v2.0",
        #         safety_checker=None,
        #         requires_safety_checker=False)
        # self.pvar.scheduler = UniPCMultistepScheduler.from_config(self.pvar.scheduler.config)
        # self.pvar.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
        # self.pvar.enable_model_cpu_offload()
        # self.pvar.enable_attention_slicing()

        def init_versatile(self):
            if self.pvd is not None: return
            print("Diffusers: Loading Versatile Diffusion...")
            from diffusers import VersatileDiffusionPipeline
            self.pvd = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion",
                                                                  safety_checker=None,
                                                                  requires_safety_checker=False)
            self.pvd.scheduler = UniPCMultistepScheduler.from_config(self.pvd.scheduler.config)
            self.pvd.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
            self.pvd.enable_model_cpu_offload()
            self.pvd.enable_attention_slicing(1)

    @plugfun()
    def init_controlnet(self, models):
        def load_model(name):
            if name == "canny": return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", file='', torch_dtype=torch.float16)
            elif name == "hed":  return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", file='', torch_dtype=torch.float16)
            elif name == "depth":  return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", file='', torch_dtype=torch.float16)
            elif name == "temporal":  return ControlNetModel.from_pretrained("CiaraRowles/TemporalNet", file='', torch_dtype=torch.float16)


        print("Diffusers: Loading ControlNet...")
        model_list = []
        for model in models:
            if model not in self.controlnets:
                self.controlnets[model] = load_model(model)
            model_list.append(self.controlnets[model])

        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.pcontrolnet = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                requires_safety_checker=False,
                controlnet=model_list,
                torch_dtype=torch.float16).to("cuda")
        self.pcontrolnet.scheduler = UniPCMultistepScheduler.from_config(self.pcontrolnet.scheduler.config)
        self.compel = Compel(tokenizer=self.pcontrolnet.tokenizer, text_encoder=self.pcontrolnet.text_encoder)

        # if self.pcontrolnet is None:
        # else:
        #     print("Swapping controlnets...")
        #     self.pcontrolnet.controlnet = StableDiffusionControlNetPipeline.from_pretrained(
        #             "runwayml/stable-diffusion-v1-5",
        #             safety_checker=None,
        #             requires_safety_checker=False,
        #             controlnet=model_list,
        #             unet = self.pcontrolnet.unet,
        #             text_encoder = self.pcontrolnet.text_encoder,
        #             vae = self.pcontrolnet.vae,
        #             tokenizer = self.pcontrolnet.tokenizer,
        #             scheduler = self.pcontrolnet.scheduler,
        #             torch_dtype=torch.float16).to("cuda")

    @plugfun()
    def init_sd(self):
        if self.ptxt is not None: return
        print("Diffusers: Loading SD...")

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.ptxt = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            vae=vae,
                                                            safety_checker=None,
                                                            requires_safety_checker=False,
                                                            torch_dtype=torch.float32).to("cuda")
        self.pimg = StableDiffusionImg2ImgPipeline(
                vae=vae,
                text_encoder=self.ptxt.text_encoder,
                tokenizer=self.ptxt.tokenizer,
                unet=self.ptxt.unet,
                scheduler=self.ptxt.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False).to("cuda")
        self.compel = Compel(tokenizer=self.ptxt.tokenizer, text_encoder=self.ptxt.text_encoder)

        # self.ptxt.enable_model_cpu_offload()
        # self.ptxt.enable_attention_slicing(1)
        # pipe.enable_vae_tiling()
        # pipe.enable_sequential_cpu_offload()
        # pipe.unet.to(memory_format=torch.channels_last)  # in-place operation


    @plugfun(plugfun_img)
    def txt2img_cn(self,
                   prompt: str = None,
                   negprompt: str = None,
                   cfg: float = 7.0,
                   image: str = None,
                   ccg: float = 1.0,
                   chg: float = 0.01,
                   steps: int = 30,
                   seed: int = 0,
                   w: int = 512,
                   h: int = 512,
                   **kwargs):
        from rendering import renderer
        session = renderer.session

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)
        # hud(image=image)

        if not w: w = session.w
        if not h: h = session.h
        if not w and image.width: w = image.width
        if not h and image.height: h = image.height

        if isinstance(image, (list, tuple)):
            image = [load_pil(i) for i in image]
        else:
            image = load_pil(image)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, generator)
        # print(latents)
        # print(latents.dtype)
        latent_arr = latents.cpu().numpy()
        from scripts.libs import tricks
        latents = tricks.grain_mul(latent_arr, chg)
        latents = torch.from_numpy(latents).to(device).to(torch.float16)
        # print(latents)
        # print(latents.dtype)

        image = self.pcontrolnet(
                prompt=prompt,
                image=image or session.img,
                # rv.img = im.astype(np.uint8)
                width=w,
                height=h,
                guidance_scale=cfg,
                controlnet_conditioning_scale=ccg,
                num_inference_steps=int(steps),
                latents=latents).images[0]

        return pil2cv(image)

    @plugfun(plugfun_img)
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
        self.init_sd()
        from rendering import renderer
        session = renderer.session

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        image = self.ptxt(prompt=prompt,
                          image=image or session.img,
                          guidance_scale=cfg,
                          num_inference_steps=int(steps),
                          generator=generator).images[0]

        return pil2cv(image)

    @plugfun(plugfun_img)
    def img2img(self,
                prompt: str = None,
                negprompt: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                w:int = 512,
                h:int = 512,
                **kwargs):
        self.init_sd()
        # return self.txt2img(self, job)
        from rendering import renderer
        session = renderer.session

        if not w: w = session.w
        if not h: h = session.h
        if not w and image.width: w = image.width
        if not h and image.height: h = image.height

        if isinstance(image, list):
            image = [load_pil(i) for i in image]
        else:
            image = load_pil(image)

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        print(image)
        images = self.pimg(prompt=prompt,
                           image=image,
                           strength=chg,
                           guidance_scale=cfg,
                           num_inference_steps=int(steps),
                           generator=generator).images


        return pil2cv(images[0])

    @plugfun(plugfun_img)
    def vd_var(self,
               cfg: float = 7.5,
               image: str = None,
               ccg: float = 1.0,
               chg: float = 0.5,
               steps: int = 30,
               seed: int = 0,
               **kwargs):
        self.init_versatile()
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

        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        result = self.pvd.image_variation(image,
                                          width=image.width,
                                          height=image.height,
                                          negative_prompt=None,
                                          guidance_scale=cfg,
                                          num_inference_steps=int(steps),
                                          generator=generator)
        ret = result["images"][0]

        return pil2cv(ret)

