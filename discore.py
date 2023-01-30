#!/usr/bin/python3

import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

from jargs import argp, args, spaced_args

logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

os.chdir(Path(__file__).parent)

DEFAULT_ACTION = None
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
        print('----------------------------------------')
        print("\n\n")
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


def on_ctrl_c():
    from src_core.classes.logs import logdiscore
    from src_core.rendering import renderer

    logdiscore("Exiting because of Ctrl+C.")
    renderer.request_stop = True
    exit(0)


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


def main():
    from src_core import core
    from src_core.classes.logs import logdiscore_err
    from src_core.classes.logs import logdiscore
    from yachalk import chalk

    if args.newplug:
        plugin_wizard()
        return

    # Deployment
    # ----------------------------------------
    if args.local:
        from deploy import deploy_local
        deploy_local()
    elif args.vastai or args.vastai_continue:
        from deploy import deploy_vastai
        deploy_vastai()
    else:
        from src_core.classes import common
        common.setup_ctrl_c(on_ctrl_c)

        from src_core.classes.paths import parse_action_script
        a, sc = parse_action_script(args.action, DEFAULT_ACTION)

        logdiscore(chalk.green(f"action: {a}"))
        logdiscore(chalk.green(f"script: {sc}"))

        if a is not None:
            # Run an action script
            # ----------------------------------------
            # Get the path, check if it exists
            apath = paths.get_script_file_path(a)
            if not apath.is_file():
                logdiscore_err(f"Unknown action '{args.action}' (searched at {apath})")
                print_possible_scripts()
                exit(1)

            # Load the action script module
            amod = importlib.import_module(paths.get_script_module_path(a), package=a)
            if amod is None:
                logdiscore_err(f"Couldn't load '{args.action}'")
                print_possible_scripts()
                exit(1)

            # By specifying this attribute, we can skip session loading when it's unnecessary to gain speed

            # t = threading.Thread(target=src_core.renderer.start_mainloop)
            # t.start()
            amod.action(args)

            print("Action done.")


            # threading.Thread(target=amod.action, args=tuple([args])).start()
            # renderer.start_mainloop()
        else:
            # Nothing is specified
            # ----------------------------------------
            core.init(pluginstall=args.install)

            # Dry run, only install and exit.
            if args.dry:
                logdiscore("Exiting because of --dry argument")
                exit(0)

            # Start server
            from src_core import server
            server.run()


def print_possible_scripts():
    from src_core.classes.logs import logdiscore
    logdiscore("All scripts: ")
    # Iterate with os.walk
    for root, dirs, files in os.walk(paths.scripts):
        files = sorted(files, key=len)
        if 'libs' not in root:
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    # Print the relative path to root without extension
                    print(f"  {os.path.relpath(os.path.join(root, file), paths.scripts)[:-3]}")

# from classes.printlib import cpuprofile
# with cpuprofile():
#     import pytorch_lightning as pl

from src_core.classes import printlib
from src_core import core
import user_conf

printlib.print_timing = user_conf.print_timing
printlib.print_trace = user_conf.print_trace
printlib.print_gputrace = user_conf.print_gputrace

if user_conf.print_extended_init:
    from src_core import installing
    installing.print_info()
    print()

core.setup_annoying_logging()

main()
