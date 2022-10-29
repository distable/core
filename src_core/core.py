import argparse
import os
import shlex
import signal
import sys

import user_conf
from src_core import paths, plugins, printlib
from src_core.installing import is_installed, pipargs, python, run
from src_core.printlib import print_info

cparse = argparse.ArgumentParser()
cparse.add_argument("--dry", action='store_true', help="Only install and test the core, do not launch server.")
cargs = None

from yachalk import chalk


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

    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()

    # CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)

    print_info()
    # Memory monitor
    # mem_mon = memmon.MemUsageMonitor("MemMon", devicelib.device, options.opts)  # TODO remove options
    # mem_mon.start()

    # 1. Prepare args
    # ----------------------------------------
    args = shlex.split(commandline_args)
    sys.argv += args
    cargs = cparse.parse_args(args)

    install_core()

    # Prepare plugin system
    # ----------------------------------------
    print()
    mprint(chalk.green_bright("1. Downloading plugins"))
    plugins.download(user_conf.install)

    print()
    mprint(chalk.green_bright("2. Initializing plugins"))

    plugins.load_urls(user_conf.startup)

    jobs = plugins.get_jobs()
    if len(jobs) > 0:
        mprint(f"Found {len(jobs)} jobs:")
        for j in jobs:
            if not user_conf.print_more:
                mprint(f" - {j.jid}")
            else:
                mprint(f" - {j.jid} ({j.func})")

    # Installations
    # ----------------------------------------

    print()
    mprint(chalk.green_bright("3. Installing plugins..."))
    plugins.broadcast("install", msg="{id}")

    # Loading plugins
    # ----------------------------------------
    print()
    mprint(chalk.green_bright("4. Loading plugins..."))
    plugins.broadcast("load", "{id}")

    print()
    mprint("All ready!")


mprint = printlib.make_print("core")
mprinterr = printlib.make_printerr("core")
torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
