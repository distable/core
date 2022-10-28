import argparse
import os
import shlex
import signal
import sys

import user_conf
from src_core import paths, plugins, printlib
from src_core.installing import is_installed, pipargs, python, run
from src_core.paths import root
from src_core.printlib import print_info

cparse = argparse.ArgumentParser()
cparse.add_argument("--dry", action='store_true', help="Only install and test the core, do not launch server.")
cparse.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
cargs = None


def install_core():
    """
    Install all core requirements
    """
    paths.plug_repos.mkdir(exist_ok=True)
    paths.plug_logs.mkdir(exist_ok=True)
    paths.plug_res.mkdir(exist_ok=True)
    paths.plugins.mkdir(exist_ok=True)
    paths.sessions.mkdir(exist_ok=True)

    if not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    try:
        import torch
        assert torch.cuda.is_available()
    except:
        mprinterr('Torch is not able to use GPU')
        sys.exit(1)

    if not is_installed("clip"):
        pipargs(f"install {clip_package}", "clip")


def sigint_handler(sig, frame):
    print(f'Interrupted with signal {sig} in {frame}')
    os._exit(0)


def init():
    global cargs

    # CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)

    print_info()
    # Memory monitor
    print()
    # mem_mon = memmon.MemUsageMonitor("MemMon", devicelib.device, options.opts)  # TODO remove options
    # mem_mon.start()
    # 1. Prepare args
    # ----------------------------------------
    args = shlex.split(commandline_args)
    sys.argv += args
    cargs = cparse.parse_args(args)

    # Prepare plugin system
    # ----------------------------------------
    plugins.download(user_conf.plugins)

    mprint("Initializing plugins")
    # Iterate all directories in paths.repodir TODO this should be handled automatically by plugin installations
    for d in paths.plug_repos.iterdir():
        sys.path.insert(0, d.as_posix())

    sys.path.insert(0, (root / "plugin-repos" / "stable_diffusion" / "ldm").as_posix())

    # TODO git clone modules from a user list
    plugins.load_urls(user_conf.startup)

    # Installations
    # ----------------------------------------
    install_core()
    mprint("Installing plugins...")
    plugins.broadcast("install", msg="{id}")

    # Loading plugins
    # ----------------------------------------
    mprint("Loading plugins...")
    plugins.broadcast("load", "{id}")
    mprint("All ready!")


mprint = printlib.make_print("core")
mprinterr = printlib.make_printerr("core")
torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
