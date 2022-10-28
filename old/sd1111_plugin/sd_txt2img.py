import math

import numpy as np
import torch
from PIL import Image

from src_plugins.sd1111_plugin import devices, images, SDState
from src_plugins.sd1111_plugin.SDJob import create_random_tensors, opt_C, opt_f, SDJob


class sd_txt2img(SDJob):
    def __init__(self,
                 enable_hr: bool = False,
                 denoising_strength: float = 0.75,
                 firstphase_width: int = 0,
                 firstphase_height: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.firstphase_width = firstphase_width
        self.firstphase_height = firstphase_height
        self.truncate_x = 0
        self.truncate_y = 0

    def init(self, model, all_prompts, all_seeds, all_subseeds):
        from src_plugins.sd1111_plugin import sd_samplers

        self.sdmodel = model
        self.sampler = sd_samplers.create_sampler(self.sampler_id, model)
        if self.enable_hr:
            # if state.job_count == -1:
            #     state.job_count = self.n_iter * 2
            # else:
            #     state.job_count = state.job_count * 2

            self.extra_generation_params["First pass size"] = f"{self.firstphase_width}x{self.firstphase_height}"

            if self.firstphase_width == 0 or self.firstphase_height == 0:
                desired_pixel_count = 512 * 512
                actual_pixel_count = self.width * self.height
                scale = math.sqrt(desired_pixel_count / actual_pixel_count)
                self.firstphase_width = math.ceil(scale * self.width / 64) * 64
                self.firstphase_height = math.ceil(scale * self.height / 64) * 64
                firstphase_width_truncated = int(scale * self.width)
                firstphase_height_truncated = int(scale * self.height)
            else:
                width_ratio = self.width / self.firstphase_width
                height_ratio = self.height / self.firstphase_height

                if width_ratio > height_ratio:
                    firstphase_width_truncated = self.firstphase_width
                    firstphase_height_truncated = self.firstphase_width * self.height / self.width
                else:
                    firstphase_width_truncated = self.firstphase_height * self.width / self.height
                    firstphase_height_truncated = self.firstphase_height

            self.truncate_x = int(self.firstphase_width - firstphase_width_truncated) // opt_f
            self.truncate_y = int(self.firstphase_height - firstphase_height_truncated) // opt_f

    def create_dummy_mask(self, x, width=None, height=None):
        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            height = height or self.height
            width = width or self.width

            # The "masked-image" in this case will just be all zeros since the entire image is masked.
            image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
            image_conditioning = self.sdmodel.get_first_stage_encoding(self.sdmodel.encode_first_stage(image_conditioning))

            # Add the fake full 1s mask to the first dimension.
            image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
            image_conditioning = image_conditioning.to(x.dtype)

        else:
            # Dummy zero conditioning if we're not using inpainting model.
            # Still takes up a bit of memory, but no encoder call.
            # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
            image_conditioning = torch.zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

        return image_conditioning

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        from src_plugins.sd1111_plugin.SDPlugin import decode_first_stage

        if not self.enable_hr:
            x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
            samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.create_dummy_mask(x))
            return samples

        x = create_random_tensors([opt_C, self.firstphase_height // opt_f, self.firstphase_width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.create_dummy_mask(x, self.firstphase_width, self.firstphase_height))

        samples = samples[:, :, self.truncate_y // 2:samples.shape[2] - self.truncate_y // 2, self.truncate_x // 2:samples.shape[3] - self.truncate_x // 2]

        if SDState.use_scale_latent_for_hires_fix:
            samples = torch.nn.functional.interpolate(samples, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

        else:
            decoded_samples = decode_first_stage(self.sdmodel, samples)
            lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                image = images.resize_image(0, image, self.width, self.height)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(devices.device)
            decoded_samples = 2. * decoded_samples - 1.

            samples = self.sdmodel.get_first_stage_encoding(self.sdmodel.encode_first_stage(decoded_samples))

        # SDPlugin.state.nextjob()
        # self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers, self.sampler_index, self.sd_model)

        noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        # GC now before running the next img2img to prevent running out of memory
        x = None
        devices.torch_gc()

        samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps, image_conditioning=self.create_dummy_mask(samples))

        return samples