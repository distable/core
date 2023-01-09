import hashlib
import os
import time

import k_diffusion as K
import numpy as np
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn
from torchvision.transforms import functional as TF

from src_core import plugins
from src_core.classes.Plugin import Plugin
from src_core.classes.prompt_job import prompt_job
from src_core.installing import run
from src_core.lib import devices
from src_core.plugins import get_plug, plugjob

SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

class kup(prompt_job):
    def __init__(self,
                 num_samples=1,
                 batch_size=1,
                 decoder='finetuned_840k', # [finetuned_840k, finetuned_560k]
                 cfg=1, # 0 to 10
                 noise_aug_level=0, # 0 to 0.6
                 noise_aug_type='gaussian', # [gaussian, fake]
                 sampler='dpm-adaptive', # [k-euler, k-euler-ancestral, k-dpm-2-ancestral, k-dpm-fast, k-dpm-adaptive]
                 steps=50, # 50
                 tol_scale=0.25, # 0.25
                 eta=1.0, # 1.0
                 **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.decoder = decoder
        self.cfg = cfg
        self.noise_aug_level = noise_aug_level
        self.noise_aug_type = noise_aug_type
        self.sampler = sampler
        self.steps = steps
        self.tol_scale = tol_scale
        self.eta = eta

def fetch(url_or_path):
    if url_or_path.startswith('http:') or url_or_path.startswith('https:'):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode('utf-8')).hexdigest()
        cachename = f'{cachekey}{ext}'
        if not os.path.exists(f'cache/{cachename}'):
            os.makedirs('tmp', exist_ok=True)
            os.makedirs('cache', exist_ok=True)
            run(f"curl -L '{url_or_path}' > tmp/{cachename}")
            os.rename(f'tmp/{cachename}', f'cache/{cachename}')
        return f'cache/{cachename}'
    return url_or_path


def make_upscaler_model(config_path, model_path, pooler_dim=768, train=False, device='cpu'):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
            model,
            sigma_data=config['model']['sigma_data'],
            embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)

class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1., embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode='nearest') * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(input, sigma, unet_cond=low_res_in, mapping_cond=mapping_cond, cross_cond=cross_cond, cross_cond_padding=cross_cond_padding, **kwargs)

def download_from_huggingface(repo, filename):
    from requests.exceptions import HTTPError
    import huggingface_hub
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename)
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(f'Go here and agree to the click through license on your account: https://huggingface.co/{repo}')
                input('Hit enter when ready:')
                continue
            else:
                raise e


def load_model_from_config(config, ckpt):
    print(f"{ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(get_plug('kupscale').repo(config))
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to(devices.cpu).eval().requires_grad_(False)
    return model


class KupscalePlugin(Plugin):
    def title(self):
        return "kupscale"

    def unload(self):
        pass

    def uninstall(self):
        pass
    def describe(self):
        return "The new StableDiffusion upscalers by RiversHaveWings"

    def init(self):
        pass

    def install(self):
        self.vae_840k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt")
        self.vae_560k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt")

    def load(self):
        model_up = make_upscaler_model(fetch('https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'),
                                       fetch('https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'))

        # sd_model = load_model_from_config("stable-diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path)
        vae_model_840k = load_model_from_config("latent_diffusion/models/first_stage_models/kl-f8/config.yaml", self.vae_840k_model_path)
        vae_model_560k = load_model_from_config("latent_diffusion/models/first_stage_models/kl-f8/config.yaml", self.vae_560k_model_path)

        # sd_model = sd_model.to(device)
        self.vae_model_840k = vae_model_840k.to(devices.device)
        self.vae_model_560k = vae_model_560k.to(devices.device)
        self.model_up = model_up.to(devices.device)

        self.tok_up = CLIPTokenizerTransform()
        self.text_encoder_up = CLIPEmbedder(device=devices.device)

    @plugjob
    def kup(self, j:kup):
        self.run(None, j)

    @torch.no_grad()
    def condition_up(self, prompts):
        return self.text_encoder_up(self.tok_up(prompts))

    @torch.no_grad()
    def run(self, seed, j:kup):
        timestamp = int(time.time())
        if not seed:
            print('No seed was provided, using the current time.')
            seed = timestamp
        print(f'upscaling with seed={seed}')
        seed_everything(seed)

        uc = self.condition_up(j.batch_size * [""])
        c = self.condition_up(j.batch_size * [j.prompt])

        if j.decoder == 'finetuned_840k':
            vae = self.vae_model_840k
        elif j.decoder == 'finetuned_560k':
            vae = self.vae_model_560k
        else:
            raise ValueError(f"Unknown decoder: {j.decoder}")

        # image = Image.open(fetch(input_file)).convert('RGB')
        image = j.ctx.image.convert('RGB')
        image = TF.to_tensor(image).to(devices.device) * 2 - 1
        low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
        low_res_decoded = vae.decode(low_res_latent/SD_Q)

        [_, C, H, W] = low_res_latent.shape

        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(self.model_up, uc, cond_scale=j.cfg)
        low_res_sigma = torch.full([j.batch_size], j.noise_aug_level, device=devices.device)
        x_shape = [j.batch_size, C, 2*H, 2*W]

        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
            sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), j.steps+1).exp().to(devices.device)
            if j.sampler == 'euler':
                return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
            elif j.sampler == 'euler-a':
                return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=j.eta)
            elif j.sampler == 'dpm2-a':
                return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=j.eta)
            elif j.sampler == 'dpm-fast':
                return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, j.steps, extra_args=extra_args, eta=j.eta)
            elif j.sampler == 'dpm-adaptive':
                sampler_opts = dict(s_noise=1., rtol=j.tol_scale * 0.05, atol=j.tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
                return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=j.eta, **sampler_opts)

        ret = []
        for _ in range((j.num_samples-1)//j.batch_size + 1):
            if j.noise_aug_type == 'gaussian':
                latent_noised = low_res_latent + j.noise_aug_level * torch.randn_like(low_res_latent)
            elif j.noise_aug_type == 'fake':
                latent_noised = low_res_latent * (j.noise_aug_level ** 2 + 1)**0.5
            else:
                raise ValueError(f"Unknown noise augmentation type: {j.noise_aug_type}")

            extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
            noise = torch.randn(x_shape, device=devices.device)
            up_latents = do_sample(noise, extra_args)

            pixels = vae.decode(up_latents/SD_Q) # equivalent to sd_model.decode_first_stage(up_latents)
            pixels = pixels.add(1).div(2).clamp(0,1)

            for j in range(pixels.shape[0]):
                img = TF.to_pil_image(pixels[j])
                ret.append(img)

        return ret


# sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in)

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_length=True, return_overflowing_tokens=False,
                                 padding='max_length', return_tensors='pt')
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging
        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(devices.device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(input_ids=input_ids.to(self.device), output_hidden_states=True)
        return clip_out.hidden_states[-1], cross_cond_padding.to(self.device), clip_out.pooler_output



