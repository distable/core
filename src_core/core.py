# The main core
#
# Has a global session state
# Has procedure to install plugins
# Can dispatch jobs
# Can deploy to cloud and connect as a client to defer onto
# ----------------------------------------

import os
import shutil
import subprocess
import sys

import socketio
from yachalk import chalk

import user_conf
from src_core import installing, jobs, plugins
from src_core.classes import paths, printlib
from src_core.classes.common import setup_ctrl_c
from src_core.classes.Job import Job
from src_core.classes.JobArgs import JobArgs
from src_core.classes.logs import logcore, logcore_err
from src_core.classes.MemMon import MemMon
from src_core.classes.paths import get_max_leadnum
from src_core.classes.PipeData import PipeData
from src_core.classes.Session import Session
from src_core.installing import is_installed, pipargs, print_info, python
from src_core.lib import devices
from src_core.classes.printlib import trace

gs: Session | None = None
printlib.print_trace = user_conf.print_trace
printlib.print_gputrace = user_conf.print_gputrace

memmon: MemMon = None

torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")

deployed = False
cclient = None


class CloudClient:
    def __init__(self):
        sio = socketio.Client()

        @sio.event
        def connect():
            pass

        @sio.event
        def disconnect():
            pass

        self.sio = sio

    def emit(self, *args, **kwargs):
        self.sio.emit(*args, **kwargs)


# region Initialization
def setup_annoying_logging():
    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")


def setup_memmon():
    global memmon
    mem_mon = MemMon("MemMon", devices.device)
    mem_mon.start()


def init(step=2, restore: bool | str | float = None, pluginstall=True):
    """
    Initialize the core and all plugins
    Args:
        pluginstall:
        step: The initialization step to stop at.
        restore:
            - False: Don't restore
            - True: Restore the latest session
            - str: Restore the session with the given name
            - float: Restore the most recent session if it's age is less than the given seconds, otherwise a new session.
    """
    global gs

    os.chdir(paths.root.as_posix())

    if isinstance(restore, bool) and restore:
        gs = Session.recent_or_now()
    elif isinstance(restore, str):
        gs = open(restore)
    elif isinstance(restore, float):
        gs = Session.recent_or_now(restore)
    else:
        gs = Session.now()

    if step >= 0:
        setup_annoying_logging()
        setup_ctrl_c()
        # setup_memmon()

        if user_conf.print_extended_init:
            print_info()
            if memmon is not None:
                print("Memory:", memmon.data['min_free'])
            print()

    if step >= 1:
        install_core()
        # pluginstall=True
        if pluginstall:
            download_plugins()
        create_plugins(pluginstall)

    if step >= 2:
        # log_jobs()
        if pluginstall:
            install_plugins()
        load_plugins()

        if user_conf.print_extended_init:
            print()
        logcore("READY")


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


def create_plugins(install=True):
    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("2. Initializing plugins"))
    plugins.instantiate_plugins_by_url([pdef.url for pdef in user_conf.plugins.values()], install=install)


def install_plugins():
    if user_conf.print_extended_init:
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

    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("Unloading plugins..."))
    plugins.broadcast("unload", "{id}", threaded=True, on_after=on_after)


def load_plugins():
    if user_conf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Loading plugins..."))
    plugins.load()


def log_jobs():
    jobs = plugins.get_jobs()
    if len(jobs) > 0:
        logcore(f"Found {len(jobs)} jobs:")
        for j in jobs:
            strjid = str(j.jid)
            if j.is_alias:
                strjid = chalk.dim(strjid)
            if not user_conf.print_more2:
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

def run0(jquery: str | JobArgs, session: Session | None = None, fg=True, **kwargs):
    if session is None:
        session = gs
    if session.ctx.image is None:
        run(jquery, session, fg=fg, **kwargs)
        add()


def run(jquery: str | JobArgs, session: Session | None = None, fg=True, **kwargs):
    """
    Run a job in the context of a session.
    """
    ifo = plugins.get_job(jquery)
    if session is None:
        session = gs
    if ifo is None:
        logcore_err(f"Job {jquery} not found!")
        return None

    # Save outputs
    def on_done(ret):
        dat = PipeData.automatic(ret)
        session.ctx.apply(dat)


    # Apply memorized session kwargs
    j = plugins.new_job(jquery, **{**session.get_kwargs(ifo), **kwargs})
    j.session = session
    j.ctx = session.ctx
    j.on_done = on_done

    # Store the prompt into ctx data
    if j.args.prompt: session.ctx.prompt = j.args.prompt
    if j.args.w: session.ctx.w = j.args.w
    if j.args.h: session.ctx.h = j.args.h

    session.add_kwargs(ifo, plugins.get_args(jquery, kwargs, True))
    session.add_job(j)

    # run
    # ----------------------------------------
    if deployed:
        cclient.emit("start_job", plugins.new_args())
    else:
        if fg:
            # logcore(f"{chalk.blue(j.jid)}(...)")
            ret = jobs.run(j)

            jargs = j.args
            jargs_str = {k: v for k, v in jargs.__dict__.items() if isinstance(v, (int, float, str))}
            jargs_str = ' '.join([f'{chalk.green(k)}={chalk.white(printlib.str(v))}' for k, v in jargs_str.items()])

            logcore(f"{chalk.blue(j.jid)}({jargs_str}) -> {chalk.grey(printlib.str(ret))}")
            return ret
        else:
            return jobs.enqueue(j)


def abort(uid):
    if deployed:
        cclient.emit("abort", uid)


def open(session_name, i=None):
    global gs
    if isinstance(session_name, Session):
        gs = session_name
    elif session_name is not None:
        gs = Session(session_name)

    return gs


def opensub(name, i=None):
    return open(gs.subsession(name), i)


def open0(session_name):
    return open(session_name, 1)


def seek(i=None):
    gs.seek(i)


def seek_next(i=1):
    gs.seek_next(i)




def seek_min():
    gs.seek_min()


def seek_max():
    gs.seek_max()


def write():
    """
    Save the current global session context data over the current path.
    """
    gs.save()


def add():
    """
    Save the current global session context data by appending.
    """
    gs.save_add()


def save():
    """
    Save the current global session context data by saving over the current context data.
    """
    gs.save()


def deploy_local():
    # A test 'provider' which attempts to do a clean clone of the current installation
    # 1. Clone the repo to ~/discore_deploy/
    subprocess.run(["git", "clone", "https://github.com/distable/core", "~/discore_deploy"])

    # 2. Copy the current config to ~/discore_deploy/user_conf.py
    shutil.copyfile("user_conf.py", "~/discore_deploy/user_conf.py")

    # 3. Copy the current project (the py file being run) to ~/discore_deploy/project.py
    shutil.copyfile(__file__, "~/discore_deploy/project.py")

    # 3. Run ~/discore_deplay/run.sh or .bat on windows
    if sys.platform == "win32":
        subprocess.run(["~/discore_deploy/run.bat"])
    else:
        subprocess.run(["~/discore_deploy/run.sh"])


def deploy_vastai():
    """
    Deploy onto cloud.
    """
    # 1. List the available machines with vastai api
    # 2. Prompt the user to choose one
    # 3. Connect to it with SSH
    # 4. Git clone the core repository
    # 5. Upload our user_conf
    # 6. Launch discore
    # 7. Connect our core to it
    # ----------------------------------------
    pass

def print_frame_header():
    from src_core import core
    print("")
    print(f"Frame {core.f} ----------------------------------------")

def __getattr__(name):
    if name == 'f':
        return gs.f
    elif name == 'image':
        return gs.image

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
