from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from src_plugins.sd1111_plugin import devices, images, masking
from src_plugins.sd1111_plugin.SDJob import create_random_tensors, opt_C, opt_f, SDJob


class SD_img2img(SDJob):
    def __init__(self,
                 init_images: list = None,
                 resize_mode: int = 0,
                 chg: float = 0.75,
                 mask: Any = None,
                 mask_blur: int = 4,
                 inpainting_fill: int = 0,
                 inpaint_full_res: bool = True,
                 inpaint_full_res_padding: int = 0,
                 mask_invert: int = 0,
                 **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = chg
        self.init_latent = None
        self.image_mask = mask
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.mask_invert = mask_invert
        self.mask = None
        self.nmask = None
        self.image_conditioning = None

    def init(self, model, all_prompts, all_seeds, all_subseeds):
        import sd_samplers
        self.sd_model = model
        self.sampler = sd_samplers.create_sampler(self.sampler_id, model)

        crop_region = None

        if self.image_mask is not None:
            self.image_mask = self.image_mask.convert('L')

            if self.mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)

            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
            else:
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(devices.device)

        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))

        if self.image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(devices.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(devices.device).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            if self.image_mask is not None:
                conditioning_mask = np.array(self.image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
            else:
                conditioning_mask = torch.ones(1, 1, *image.shape[-2:])

            # Create another latent image, this time with a masked version of the original input.
            conditioning_mask = conditioning_mask.to(image.device)
            conditioning_image = image * (1.0 - conditioning_mask)
            conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(conditioning_image))

            # Create the concatenated conditioning tensor to be fed to `c_concat`
            conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=self.init_latent.shape[-2:])
            conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
            self.image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
            self.image_conditioning = self.image_conditioning.to(devices.device).type(self.sd_model.dtype)
        else:
            self.image_conditioning = torch.zeros(
                    self.init_latent.shape[0], 5, 1, 1,
                    dtype=self.init_latent.dtype,
                    device=self.init_latent.device
            )

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        del x
        devices.torch_gc()

        return samples