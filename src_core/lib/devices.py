import contextlib
from typing import Tuple

import psutil as psutil
import torch


# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
def get_available_vram():
    if device.type == 'cuda':
        stats = torch.cuda.memory_stats(device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    else:
        return psutil.virtual_memory().available


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]: return args[x + 1]
    return None


def get_optimal_device(device_id=None):
    if torch.cuda.is_available():
        if device_id is not None:
            cuda_device = f"cuda:{device_id}"
            return torch.device(cuda_device)
        else:
            return torch.device("cuda")

    if has_mps:
        return torch.device("mps")

    return cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = True


def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32:
        return contextlib.nullcontext()
    elif dtype == torch.float16:
        return torch.autocast("cuda")

    raise ValueError(f"Unknown precision {precision}")


has_mps = getattr(torch, 'has_mps', False)
cpu = torch.device("cpu")

# State values
device = get_optimal_device()
precision = 'half'
dtype = torch.float16
dtype_vae = torch.float16


def set_optimal(_precision='half', _precision_vae='half'):
    set(get_optimal_device(), _precision, _precision_vae)


def set(_device=None, _precision='half', _precision_vae='half'):
    global device, precision, dtype, dtype_vae
    if _device is None:
        _device = get_optimal_device()

    _device = _device
    precision = _precision

    if precision == 'half':
        dtype = torch.float16
    elif precision == 'full':
        dtype = torch.float32

    if _precision_vae == 'half':
        dtype_vae = torch.float16
    elif _precision_vae == 'full':
        dtype_vae = torch.float32


# might move elsewhere
xformers_available = False

try:
    import xformers.ops

    xformers_available = True
except Exception:
    # print("Cannot import xformers", file=sys.stderr)
    # print(traceback.format_exc(), file=sys.stderr)
    pass


def get_vram_nvidia_smi() -> Tuple[int, int]:
    # Invoke nvidia-smi
    import subprocess
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
    total = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"])

    # e.g.
    # used=b'1116\n'
    # total=b'11178\n'

    # Convert to int
    used = int(used.decode('utf-8').strip())
    total = int(total.decode('utf-8').strip())

    return used,total
