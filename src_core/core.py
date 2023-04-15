# The main core
#
# Has a global session state
# Has procedure to install plugins
# Can dispatch jobs
# Can deploy to cloud and connect as a client to defer onto
# ----------------------------------------

import os
import sys

from yachalk import chalk

import jargs
import userconf
from src_core import installing, plugins
from src_core.classes import paths, printlib
from src_core.classes.common import setup_ctrl_c
from src_core.classes.logs import logcore, logcore_err
from src_core.classes.Session import Session
from src_core.installing import is_installed, pipargs, print_info, python

torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")

proxied = False
proxy = None


# class Proxy:
#     def __init__(self):
#         sio = socketio.Client()
#
#         @sio.event
#         def connect():
#             pass
#
#         @sio.event
#         def disconnect():
#             pass
#
#         self.sio = sio
#
#     def emit(self, *args, **kwargs):
#         self.sio.emit(*args, **kwargs)


# region Initialization
def setup_annoying_logging():
    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")


# def setup_memmon():
#     global memmon
#     mem_mon = MemMon("MemMon", devices.device)
#     mem_mon.start()


def init(step=2, pluginstall=None):
    """
    Initialize the core and all plugins
    Args:
        pluginstall:
        step: The initialization step to stop at.
    """
    print("core.init")

    if pluginstall is None:
        pluginstall = jargs.args.install

    os.chdir(paths.root.as_posix())

    if step >= 0:
        print("core.init(step=0)")
        # setup_annoying_logging()
        # setup_ctrl_c()

    if step >= 1:
        print("core.init(step=1)")
        install_core()
        # pluginstall=True
        if pluginstall:
            download_plugins()
        create_plugins(pluginstall)

    if step >= 2:
        print("core.init(step=2)")
        # log_jobs()
        if pluginstall:
            install_plugins()
        load_plugins()

        if userconf.print_extended_init:
            print()
        logcore("READY")
        print("")

    return Session.now(log=False)


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
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("1. Downloading plugins"))
    plugins.download_git_urls([pdef.url for pdef in userconf.plugins.values()])


def create_plugins(install=True):
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("2. Initializing plugins"))
    # plugins.instantiate_plugins_by_pids([pdef.url for pdef in userconf.plugins.values()], install=install)
    # plugins.instantiate_plugins_in(paths.plugins, install=install)


def install_plugins():
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Installing plugins..."))

    def before(plug):
        installing.default_basedir = paths.plug_repos / plug.short_pid

    def after(plug):
        installing.default_basedir = None

    plugins.broadcast("install", "{id}", on_before=before, on_after=after)


def unload_plugins():
    def on_after(plug):
        plug.loaded = False

    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("Unloading plugins..."))
    plugins.broadcast("unload", "{id}", threaded=True, on_after=on_after)


def load_plugins():
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Loading plugins..."))
    plugins.load(userconf_only=False)


def log_jobs():
    jobs = plugins.get_jobs()
    if len(jobs) > 0:
        logcore(f"Found {len(jobs)} jobs:")
        for j in jobs:
            strjid = str(j.jid)
            if j.is_alias:
                strjid = chalk.dim(strjid)
            if not userconf.print_more2:
                logcore(" -", strjid)
            else:
                logcore(" -", f"{strjid} ({j.func})")


# endregion

# def run(jquery: str | JobArgs, fg=True, **kwargs):
#     """
#     Run a job in the context of a session.
#     """
#     ifo = plugins.get_job(jquery)
#
#     if deployed:
#         cclient.emit("start_job", plugins.new_args())
#     else:
#         if fg:
#             return jobs.run(ifo)
#         else:
#             return jobs.enqueue(ifo)

# def abort(uid):
#     if proxied:
#         proxy.emit("abort", uid)


# def open(session_name, i=None)->Session:
#     global gs
#     if isinstance(session_name, Session):
#         gs = session_name
#     elif session_name is not None:
#         gs = Session(session_name)
#
#     return gs


# def opensub(name, i=None)->Session:
#     return open(gs.subsession(name), i)
#
#
# def open0(session_name):
#     return open(session_name, 1)
#
#
# def seek(i=None):
#     gs.seek(i)
#
#
# def seek_next(i=1, log=True):
#     gs.seek_next(i, log=log)
#
#
# def seek_new(log=True):
#     gs.seek_new(log=log)
#
#
# def seek_min():
#     gs.seek_min()
#
#
# def seek_max():
#     gs.seek_max()


# def write():
#     """
#     Save the current global session context data over the current path.
#     """
#     gs.save()
#
#
# def save():
#     """
#     Save the current global session context data by saving over the current context data.
#     """
#     gs.save()


# def print_frame_header():
#     from src_core import core
#     print("")
#     print(f"Frame {core.f} ----------------------------------------")
#
#
# def __getattr__(name):
#     if name == 'f':
#         return gs.f
#     elif name == 'image':
#         return gs.image
#
#     raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
