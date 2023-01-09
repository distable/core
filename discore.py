#!/usr/bin/python3

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from src_core.classes import paths

DEFAULT_ACTION = 'render'
VENV_DIR = "venv"

argp = argparse.ArgumentParser()

# Add positional argument 'project' for the script to run. optional, 'project' by default
argp.add_argument("session", nargs="?", default=None, help="Session or script")
argp.add_argument("action", nargs="?", default=None, help="Script or action to run")
argp.add_argument("subdir", nargs="?", default='', help="Subdir in the session")

argp.add_argument("--dry", action="store_true")
argp.add_argument("--newplug", action="store_true")
argp.add_argument("--recreate-venv", action="store_true")
argp.add_argument('--upgrade', action='store_true', help='Upgrade to latest version')
argp.add_argument('--install', action='store_true', help='Install plugins requirements and custom installations.')
argp.add_argument('--deploy', type=str, default='none', help='Deploy to provider')
argp.add_argument('--music', type=str, default=None, help='Music file to play in video export')
argp.add_argument('--music_start', type=float, default=0, help='Music start time in seconds')
argp.add_argument('--fps', type=int, default=30, help='FPS')
argp.add_argument('--frames', type=str, default=None, help='The frames to render in first:last format')
argp.add_argument('--w', type=int, default=None, help='The target width.')
argp.add_argument('--h', type=int, default=None, help='The target height.')
argp.add_argument('--mpv', action='store_true', help='Open video in MPV.')
argp.add_argument('--run', action='store_true', help='Perform the run in a subprocess')
# Video argument creates a video for the session name
args = argp.parse_args()

# Eat up arguments
original_args = sys.argv[1:]
sys.argv = [sys.argv[0]]

os.chdir(Path(__file__).parent)

# The actual script when launched as standalone
# ----------------------------------------

if not args.run:
    # Disallow running as root
    if os.geteuid() == 0:
        print("Please do not run as root")
        exit(1)

    # Check that we're running python 3.9 or higher
    if sys.version_info < (3, 9):
        print("Please run with python 3.9 or higher")
        exit(1)

    # Checkb that we have git installed
    if not os.path.exists("/usr/bin/git"):
        print("Please install git")
        exit(1)

    # Create a virtual environment if it doesn't exist
    if not os.path.exists(VENV_DIR):
        # venv.create(venv_dir, with_pip=True)
        os.system(f"python3 -m venv {VENV_DIR}")
        argp.upgrade = True

    # Run bash shell with commands to activate the virtual environment and run the launch script
    spaced_args = ' '.join([f'"{arg}"' for arg in original_args])
    os.system(f"bash -c 'source {VENV_DIR}/bin/activate && python3 {__file__} {spaced_args} --run'")
    exit(0)


# ----------------------------------------

# from src_core import renderer
# from src_core.renderer import get_script_path, parse_action_script, render_frame, render_init, render_loop
# from src_core import core
# from src_core.classes import paths
# from src_core.classes.logs import loglaunch, loglaunch_err
# from src_core.classes.Session import Session

def determine_session():
    return args.session or args.action or args.script


def on_ctrl_c():
    from src_core.classes.logs import loglaunch
    loglaunch("Exiting because of Ctrl+C.")
    exit(0)


def main():
    from src_core import core
    from src_core import renderer
    renderer.args = args

    if args.newplug:
        plugin_wizard()
        return

    if args.upgrade:
        # Install requirements with venv pip
        os.system(f"{VENV_DIR}/bin/pip install -r requirements.txt")
        print('Upgrading to latest version')

    if args.deploy == 'none':
        from src_core.classes import common
        common.setup_ctrl_c(on_ctrl_c)

        from src_core.classes.paths import parse_action_script
        a, sc = parse_action_script(args.action, DEFAULT_ACTION)

        if a is not None:
            # Run an action script
            # ----------------------------------------
            # Get the path, check if it exists
            apath = paths.get_script_file_path(a)
            if not apath.is_file():
                from src_core.classes.logs import loglaunch_err
                loglaunch_err(f"Unknown action '{args.action}' (searched at {apath})")
                exit(1)

            # Load the action script module
            amod = importlib.import_module(paths.get_script_module_path(a), package=a)
            if amod is None:
                from src_core.classes.logs import loglaunch_err
                loglaunch_err(f"Couldn't load '{args.action}'")
                exit(1)

            # By specifying this attribute, we can skip session loading when it's unnecessary to gain speed
            if not hasattr(amod, 'disable_startup_session') or not amod.disable_startup_session:
                core.open(determine_session())

            amod.action(args)
        else:
            # Nothing is specified
            # ----------------------------------------
            from src_core.classes.Session import Session
            core.init(restore=Session.now())

            # Dry run, only install and exit.
            if args.dry:
                from src_core.classes.logs import loglaunch
                loglaunch("Exiting because of --dry argument")
                exit(0)

            # Start server
            from src_core import server
            server.run()

    # Deployment
    # ----------------------------------------
    if args.deploy == 'local':
        # A test deployment to local machine (new installation with the configuration ready to run)
        core.deploy_local()
    elif args.deploy == 'run':
        # Run the deployment setup requested by the host (action, script, etc.)
        pass
    elif args.deploy == 'vast':
        # Ask user for vast credentials, choose a server, and deploy to it
        core.deploy_vastai()  # TODO Options to keep the instance open
    else:
        print('Unknown deployment provider.')
        exit(1)


def plugin_wizard():
    import shutil
    from src_core import installing
    import re
    from src_core.classes import paths
    from art import text2art

    PLUGIN_TEMPLATE_PREFIX = ".template"

    print(text2art("Plugin Wizard"))

    # Find template directories (start with .template)
    templates = []
    for d in paths.code_plugins.iterdir():
        if d.name.startswith(PLUGIN_TEMPLATE_PREFIX):
            templates.append(d)

    template = None

    if len(templates) == 0:
        print("No templates found.")
        exit(1)
    elif len(templates) == 1:
        template = templates[0]
    else:
        for i, path in enumerate(templates):
            s = path.name[len(PLUGIN_TEMPLATE_PREFIX):]
            while not s[0].isdigit() and not s[0].isalpha():
                s = s[1:]
            print(f"{i + 1}. {s}")

        print()
        while template is None:
            try:
                v = int(input("Select a template: ")) - 1
                if v >= 0:
                    template = templates[v]
            except:
                pass

    pid = input("ID name: ")

    clsdefault = f"{pid.capitalize()}Plugin"
    cls = input(f"Class name (default={clsdefault}): ")
    if not cls:
        cls = clsdefault

    plugdir = paths.code_plugins / f"{pid}_plugin"
    clsfile = plugdir / f"{cls}.py"

    shutil.copytree(template.as_posix(), plugdir)
    shutil.move(plugdir / "TemplatePlugin.py", clsfile)

    # Find all {{word}} with regex and ask for a replacement
    regex = r'__(\w+)__'
    with open(clsfile, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        matches = re.findall(regex, line)
        if matches:
            vname = matches[0]

            # Default values
            vdefault = ''
            if vname == 'classname': vdefault = cls
            if vname == 'title': vdefault = pid

            # Ask for a value
            if vdefault:
                value = input(f"{vname} (default={vdefault}): ")
            else:
                value = input(f"{vname}: ")

            if not value and vdefault:
                value = vdefault

            # Apply the value
            lines[i] = re.sub(regex, value, line)

    # Write lines back to file
    with open(clsfile, "w") as f:
        f.writelines(lines)

    # Open plugdir in the file explorer
    installing.open_explorer(plugdir)
    print("Done!")
    input()
    exit(0)


def framerange():
    if args.frames:
        ranges = args.frames.split('-')
        for r in ranges:
            yield r
    else:
        yield args.frames


main()
