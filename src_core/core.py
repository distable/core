import os
import sys

from yachalk import chalk

import user_conf
from src_core import installing, jobs, plugins
from src_core.classes import paths, printlib
from src_core.classes.common import setup_ctrl_c
from src_core.classes.JobArgs import JobArgs
from src_core.classes.logs import logcore, logcore_err
from src_core.classes.MemMon import MemMon
from src_core.classes.Session import Session
from src_core.installing import is_installed, pipargs, print_info, python
from src_core.lib import devices

gsession = Session.now_or_recent()
printlib.print_timing = user_conf.print_timing

memmon: MemMon = None

torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")


def setup_annoying_logging():
    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()


def setup_memmon():
    global memmon
    mem_mon = MemMon("MemMon", devices.device)
    mem_mon.start()


def init(step=2, autosave=True):
    if step >= 0:
        setup_annoying_logging()
        setup_ctrl_c()
        setup_memmon()

        if user_conf.print_extended_init:
            print_info()
            if memmon is not None:
                print("Memory:", memmon.data['min_free'])
            print()

    if step >= 1:
        install_core()
        download_plugins()
        create_plugins()

    if step >= 2:
        # log_jobs()
        install_plugins()
        load_plugins()

        if user_conf.print_extended_init:
            print()
        logcore("READY")

    gsession.autosave = autosave


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
        installing.run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    try:
        import torch
        assert torch.cuda.is_available()
    except:
        logcore_err('Torch is not able to use GPU')
        sys.exit(1)

    if not is_installed("clip"):
        pipargs(f"install {clip_package}", "clip")


def download_plugins():
    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("1. Downloading plugins"))
    plugins.download([pdef.url for pdef in user_conf.plugins.values()])


def create_plugins():
    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("2. Initializing plugins"))
    plugins.create_plugins_by_url([pdef.url for pdef in user_conf.plugins.values()])


def install_plugins():
    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Installing plugins..."))
    plugins.broadcast("install")


def unload_plugins():
    def on_after(plug):
        plug.loaded = False

    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("Unloading plugins..."))
    plugins.broadcast("unload", "{id}", threaded=True, on_after=on_after)


def load_plugins():
    def on_after(plug):
        plug.loaded = True

    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Loading plugins..."))
    plugins.broadcast("load", "{id}", threaded=False, on_after=on_after)  # TODO threaded true is weird


def log_jobs():
    jobs = plugins.get_jobs()
    if len(jobs) > 0:
        logcore(f"Found {len(jobs)} jobs:")
        for j in jobs:
            strjid = str(j.jid)
            if j.alias:
                strjid = chalk.dim(strjid)
            if not user_conf.print_more2:
                logcore(" -", strjid)
            else:
                logcore(" -", f"{strjid} ({j.func})")


def job(jquery: str | JobArgs, session=None, bg=False, **kwargs):
    """
    Run a job in the current session.
    """
    if session is None:
        session = gsession

    def on_output(dat):
        if session.autosave:
            session.save_next(dat)
        session.context = dat

    ifo = plugins.get_job(jquery)
    j = plugins.new_job(jquery, **{**session.get_kwargs(ifo), **kwargs})
    j.input = session.context
    j.on_output = on_output

    if hasattr(j.args, 'prompt') and j.args.prompt:
        session.context.prompt = j.args.prompt

    session.add_kwargs(ifo, plugins.get_args(jquery, kwargs, True))
    session.add_job(j)
    if bg:
        return jobs.enqueue(j)
    else:
        return jobs.run(j)

    # print("")


def save():
    gsession.save_next(gsession.context)
