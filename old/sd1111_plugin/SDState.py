import argparse

from src_plugins.sd1111_plugin import devices, safe
from src_plugins.sd1111_plugin.devices import get_optimal_device
from src_plugins.sd1111_plugin.SDAttention import SDAttention

attention = SDAttention.SPLIT_DOGGETT
always_batch_cond_uncond = False  # disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram

lowvram = False
medvram = True
parallel_processing_allowed = not lowvram and not medvram
lowram = False
precision = 'full'
no_half = True
no_half_vae = True
xformers = False
batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)

opt_channelslast = False
weight_load_location = None if lowram else "cpu"
use_scale_latent_for_hires_fix = False

parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
cmd_opts = parser.parse_args()

hnmodel = None
sdmodel = None
clipmodel = None

safe.run(devices.enable_tf32, "Enabling TF32")

def set_device_optimal(_precision='half', _precision_vae='half'):
    set_device(get_optimal_device(), _precision, _precision_vae)

def set_device(_device=None, _precision='half', _precision_vae=None):
    import torch

    _device = _device if _device is not None else get_optimal_device()
    _precision_vae = _precision_vae if _precision_vae is not None else _precision

    devices.device = _device
    devices.precision = _precision

    if precision == 'half':
        devices.dtype = torch.float16
    elif precision == 'full':
        devices.dtype = torch.float32

    if _precision_vae == 'half':
        devices.dtype_vae = torch.float16
    elif _precision_vae == 'full':
        devices.dtype_vae = torch.float32

set_device(devices.get_optimal_device(), 'full')
