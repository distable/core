import argparse
import os
import shlex
import signal
import sys

import user_conf
from src_core import plugins
from src_core.classes import paths
from src_core.classes.MemMon import MemMon
from src_core.installing import is_installed, pipargs, python, run
from src_core.lib import devices
from src_core.classes.printlib import print_info
from src_core.logs import logcore, logcore_err

from yachalk import chalk

cparse = argparse.ArgumentParser()
cparse.add_argument("--dry", action='store_true', help="Only install and test the core, do not launch server.")
cargs = None

memmon = None


def setup_annoying_logging():
    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()


def setup_ctrl_c():
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    # CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)


def setup_memmon():
    global memmon
    mem_mon = MemMon("MemMon", devices.device)
    mem_mon.start()


def init():
    setup_args()

    setup_annoying_logging()
    setup_ctrl_c()
    # setup_memmon()

    print_info()

    install_core()
    download_plugins()
    create_plugins()
    log_jobs()
    install_plugins()
    launch_plugins()

    print()
    logcore("All ready!")


def setup_args():
    global cargs
    args = shlex.split(commandline_args)
    sys.argv += args
    cargs = cparse.parse_args(args)


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
        logcore_err('Torch is not able to use GPU')
        sys.exit(1)

    if not is_installed("clip"):
        pipargs(f"install {clip_package}", "clip")


def download_plugins():
    print()
    logcore(chalk.green_bright("1. Downloading plugins"))
    plugins.download(user_conf.install)


def create_plugins():
    print()
    logcore(chalk.green_bright("2. Initializing plugins"))
    plugins.load_plugins_by_url(user_conf.startup)


def install_plugins():
    print()
    logcore(chalk.green_bright("3. Installing plugins..."))
    plugins.broadcast("install", msg="{id}")


def launch_plugins():
    print()
    logcore(chalk.green_bright("4. Loading plugins..."))
    plugins.broadcast("load", "{id}")


def log_jobs():
    jobs = plugins.get_jobs()
    if len(jobs) > 0:
        logcore(f"Found {len(jobs)} jobs:")
        for j in jobs:
            strjid = str(j.jid)
            if j.alias:
                strjid = chalk.dim(strjid)
            if not user_conf.print_more:
                logcore(" -", strjid)
            else:
                logcore(" -", f"{strjid} ({j.func})")


torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
