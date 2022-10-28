from pathlib import Path

from src_core import paths

ckpt = Path("sd-v1-4.ckpt")
config = paths.plug_repos / 'stable_diffusion' / 'configs/stable-diffusion/v1-inference.yaml'
vae_path = None

