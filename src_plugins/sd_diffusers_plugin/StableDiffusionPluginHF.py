# import torch
#
# from core.plugins import Plugin
#
# from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
#
#
# class StableDiffusionPluginHF(Plugin):
#     def txt2img(self):
#         from transformers import CLIPTextModel, CLIPTokenizer
#         from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
#
#         p = "~/stable-diffusion-webui-dev/models/Stable-diffusion/sd-v1-4.ckpt"
#         pipe = StableDiffusionPipeline.from_pretrained(p, torch_type=torch.float32, revision="fp32")
#         # pipe = pipe.to("cuda")
#
#         prompt = "a photo of an astronaut riding a horse on mars"
#         image = pipe(prompt).images[0]
#
#         # 1. Load the autoencoder model which will be used to decode the latents into image space.
#         vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
#
#         # 2. Load the tokenizer and text encoder to tokenize and encode the text.
#         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#         text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
#
#         # 3. The UNet model for generating the latents.
#         unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
#
#         scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
#
