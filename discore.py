#!/usr/bin/python3

import importlib
import os
import shutil
import sys
from pathlib import Path

from jargs import argp, args, original_args, spaced_args

os.chdir(Path(__file__).parent)

DEFAULT_ACTION = 'render'
VENV_DIR = "venv"

# The actual script when launched as standalone
# ----------------------------------------
# python_exec = sys.executable
# print(python_exec)

if not args.run:
    # Disallow running as root
    if os.geteuid() == 0:
        print("You are warning as root, proceed at your own risks")

    # Check that we're running python 3.9 or higher
    if sys.version_info < (3, 9):
        print(f"Warning your python version {sys.version_info} is detected as lower than 3.9, you may be fucked, proceed at your own risks")

    # Check that we have git installed
    if not os.path.exists("/usr/bin/git"):
        print("Please install git")
        exit(1)

    # Create a virtual environment if it doesn't exist
    if not args.no_venv:
        if not os.path.exists(VENV_DIR) or args.recreate_venv:
            if args.recreate_venv:
                shutil.rmtree(VENV_DIR)

            os.system(f"python3 -m venv {VENV_DIR}")
            argp.upgrade = True

        # Run bash shell with commands to activate the virtual environment and run the launch script
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate'")

    if args.upgrade:
        # Install requirements with venv pip
        if args.no_venv:
            os.system(f"{sys.executable} -m pip install -r requirements.txt")
        else:
            os.system(f"{VENV_DIR}/bin/pip install -r requirements.txt")
        print('Upgrading to latest version')
        exit(0)

    if args.no_venv:
        os.system(f"bash -c '{sys.executable} {__file__} {spaced_args} --upgrade --run'")
    else:
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate && python3 {__file__} {spaced_args} --upgrade --run'")

    exit(0)

# ----------------------------------------

# from src_core import renderer
# from src_core.renderer import get_script_path, parse_action_script, render_frame, render_init, render_loop
# from src_core import core
# from src_core.classes import paths
# from src_core.classes.logs import loglaunch, loglaunch_err
# from src_core.classes.Session import Session

from src_core.classes import paths


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

    # Deployment
    # ----------------------------------------
    if args.local:
        from deploy import deploy_local
        deploy_local()
    elif args.vastai:
        from deploy import deploy_vastai
        deploy_vastai()
    else:
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
            print(paths.get_script_module_path(a))
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
            core.init(pluginstall=args.install, restore=Session.now())

            # Dry run, only install and exit.
            if args.dry:
                from src_core.classes.logs import loglaunch
                loglaunch("Exiting because of --dry argument")
                exit(0)

            # Start server
            from src_core import server
            server.run()


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
