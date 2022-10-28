import random
from typing import Any, Dict

import torch

from src_plugins.sd1111_plugin import devices
from src_core.jobs import JobParams

opt_C = 4
opt_f = 8
eta_noise_seed_delta = 0


class SDJob(JobParams):
    def __init__(self,
                 prompt: str = "",
                 sampler: int = 'euler-a',
                 seed: int = -1,
                 subseed: int = -1,
                 subseed_strength: float = 0,
                 seed_resize_from_h: int = -1,
                 seed_resize_from_w: int = -1,
                 seed_enable_extras: bool = True,
                 batch_size: int = 1,
                 steps: int = 22,
                 cfg: float = 7.0,
                 width: int = 512,
                 height: int = 512,
                 tiling: bool = False,
                 extra_generation_params: Dict[Any, Any] = None,
                 overlay_images: Any = None,
                 promptneg: str = None,
                 eta: float = None,
                 ddim_discretize: str = 'uniform',  # [ 'uniform', 'quad' ]
                 s_churn: float = 0.0,
                 s_tmax: float = None,
                 s_tmin: float = 0.0,
                 s_noise: float = 1.0):
        super(SDJob, self).__init__()
        self.prompt: str = prompt
        self.promptneg: str = promptneg or ""
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.width: int = width
        self.height: int = height
        self.cfg: float = cfg
        self.sampler_id: int = sampler
        self.batch_size: int = batch_size
        self.steps: int = steps
        self.tiling: bool = tiling
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.denoising_strength: float = 0
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax or float('inf')  # not representable as a standard ui option
        self.s_noise = s_noise

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.all_prompts = None
        self.all_seeds = None
        self.all_subseeds = None

    def init(self, model, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()


def store_latent(decoded):
    # state.current_latent = decoded

    # if opts.show_progress_every_n_steps > 0 and SDPlugin.state.sampling_step % opts.show_progress_every_n_steps == 0:
    #     if not SDPlugin.parallel_processing_allowed:
    #         SDPlugin.state.current_image = sample_to_image(decoded)
    pass


def slerp(val, low, high):
    # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 or eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h // 8, seed_resize_from_w // 8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(devices.device) for n in sampler_noises]

    x = torch.stack(xs).to(devices.device)
    return x


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)
