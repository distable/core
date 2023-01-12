#!/usr/bin/python3

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src_core.lib.corelib import open_in_explorer


DEFAULT_ACTION = 'render'
VENV_DIR = "venv"

argp = argparse.ArgumentParser()

# Add positional argument 'project' for the script to run. optional, 'project' by default
argp.add_argument("session", nargs="?", default=None, help="Session or script")
argp.add_argument("action", nargs="?", default=None, help="Script or action to run")
argp.add_argument("subdir", nargs="?", default='', help="Subdir in the session")

argp.add_argument('--run', action='store_true', help='Perform the run in a subprocess')
argp.add_argument("--recreate-venv", action="store_true")
argp.add_argument("--no-venv", action="store_true")
argp.add_argument('--upgrade', action='store_true', help='Upgrade to latest version')
argp.add_argument('--install', action='store_true', help='Install plugins requirements and custom installations.')

argp.add_argument("--dry", action="store_true")
argp.add_argument("--newplug", action="store_true", help="Create a new plugin with the plugin wizard")

# Script Arguments
argp.add_argument('--fps', type=int, default=30, help='FPS')
argp.add_argument('--frames', type=str, default=None, help='The frames to render in first:last format')
argp.add_argument('--w', type=int, default=None, help='The target width.')
argp.add_argument('--h', type=int, default=None, help='The target height.')
argp.add_argument('--music', type=str, default=None, help='Music file to play in video export')
argp.add_argument('--music_start', type=float, default=0, help='Music start time in seconds')
argp.add_argument('--mpv', action='store_true', help='Open the resulting video in MPV.')

# Deployment
argp.add_argument('--deploy', type=str, default='none', help='Deploy to provider')
argp.add_argument('--shell', action='store_true', default=None, help='Open a shell in the deployed remote.')
argp.add_argument('--vastai_search', type=str, default=None, help='Search for a VastAI server')

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
        spaced_args = ' '.join([f'"{arg}"' for arg in original_args])
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate'")

    if args.upgrade:
        # Install requirements with venv pip
        if args.no_venv:
            os.system(f"pip install -r requirements.txt")
        else:
            os.system(f"{VENV_DIR}/bin/pip install -r requirements.txt")
        print('Upgrading to latest version')

    if args.no_venv:
        os.system(f"bash -c 'python3 {__file__} {spaced_args} --upgrade --run'")
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

import user_conf
from src_core.classes import paths
import interactive


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
        deploy_local()
    elif args.deploy == 'run':
        # Run the deployment setup requested by the host (action, script, etc.)
        pass
    elif args.deploy == 'vast':
        # Ask user for vast credentials, choose a server, and deploy to it
        deploy_vastai()  # TODO Options to keep the instance open
    else:
        print('Unknown deployment provider.')
        exit(1)


# User files/directories to copy at the start of a deployment
deploy_copy = ['requirements.txt', 'discore.py', paths.userconf_name, paths.scripts_name]


# Commands to run in order to setup a deployment
def get_deploy_commands(clonepath):
    return [
        ['git', 'clone', 'https://github.com/distable/core', clonepath],
        ['git', '-C', clonepath, 'submodule', 'update', '--init', '--recursive']
    ]


def deploy_local():
    import time
    import shutil
    import platform


    # A test 'provider' which attempts to do a clean clone of the current installation
    # 1. Clone the repo to ~/discore_deploy/
    src = paths.root
    dst = Path.home() / "discore_deploy"

    shutil.rmtree(dst.as_posix())

    cmds = get_deploy_commands(dst.as_posix())
    for cmd in cmds:
        subprocess.run(cmd)

    for file in deploy_copy:
        shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])


def input_bool(string):
    while True:
        yes = input(string).lower() in ['yes', 'y', 'true', 't', '1']
        no = input(string).lower() in ['no', 'n', 'false', 'f', '0']
        if yes: return True
        if no: return False

        print("Please respond with 'y' or 'n'")


def sshexec(ssh, cmd, with_printing=True):
    if with_printing:
        print(f'sshexec({cmd})')
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    ret = []
    for line in iter(stdout.readline, ""):
        # print(line, end="")
        ret.append(line)
    if with_printing:
        from yachalk import chalk
        print(chalk.dim('\n'.join(ret)))

    return ret


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
    vastpath = paths.root / 'vast'
    if not vastpath.is_file():
        if sys.platform == 'linux':
            os.system("wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;")
        else:
            print("Vastai deployment is only supported on Linux, may be broken on windows. it will probably break, good luck")

    # Run vast command and parse its output

    def fetch_offers():
        out = subprocess.check_output(['python3', vastpath, 'search', 'offers', args.vastai_search or user_conf.vastai_default_search]).decode('utf-8')
        # Example output:
        # ID       CUDA  Num  Model     PCIE  vCPUs    RAM  Disk   $/hr    DLP    DLP/$  NV Driver   Net_up  Net_down  R     Max_Days  mach_id  verification
        # 5563966  11.8  14x  RTX_3090  12.7  64.0   257.7  1672   5.8520  341.8  58.4   520.56.06   23.1    605.7     99.5  22.4      6294     verified
        # 5412452  12.0   4x  RTX_3090  24.3  32.0   257.8  751    1.6800  52.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
        # 5412460  12.0   2x  RTX_3090  24.3  16.0   257.8  376    0.8400  26.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
        # 5412453  12.0   1x  RTX_3090  24.3  8.0    257.8  188    0.4200  13.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
        # 5412458  12.0   8x  RTX_3090  24.3  64.0   257.8  1502   3.3600  103.9  30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified

        lines = out.splitlines()
        offers = list()  # list of dict
        for i, line in enumerate(lines):
            line = line.split()
            if len(line) == 0: continue
            if i == 0: continue
            offers.append(dict(id=line[0], cuda=line[1], num=line[2], model=line[3], pcie=line[4], vcpus=line[5], ram=line[6], disk=line[7], price=line[8], dlp=line[9], dlpprice=line[10], nvdriver=line[11], netup=line[12], netdown=line[13], r=line[14], maxdays=line[15], machid=line[16], verification=line[17]))

        return offers

    def fetch_instances():
        out = subprocess.check_output(['python3', vastpath, 'show', 'instances']).decode('utf-8')
        # Example output:
        # ID       Machine  Status  Num  Model     Util. %  vCPUs    RAM  Storage  SSH Addr      SSH Port  $/hr    Image            Net up  Net down  R     Label
        # 5717760  5721     -        1x  RTX_3090  -        6.9    128.7  17       ssh4.vast.ai  37760     0.2436  pytorch/pytorch  75.5    75.1      98.5  -

        lines = out.splitlines()
        instances = list()
        for i, line in enumerate(lines):
            line = line.split()
            if len(line) == 0: continue
            if i == 0: continue
            instances.append(dict(id=line[0], machine=line[1], status=line[2], num=line[3], model=line[4], util=line[5], vcpus=line[6], ram=line[7], storage=line[8], sshaddr=line[9], sshport=line[10], price=line[11], image=line[12], netup=line[13], netdown=line[14], r=line[15], label=line[16]))
        return instances

    from yachalk import chalk

    def print_offer(e):
        print(chalk.green(f'{i + 1} - {e["model"]} - {e["num"]} - {e["dlp"]} - {e["price"]} $/hr - {e["dlpprice"]} DLP/HR'))

    def print_instance(e):
        print(chalk.green_bright(f'{i + 1} - {e["model"]} - {e["num"]} -  {e["price"]} $/hr - {e["status"]}'))

    instances = fetch_instances()
    selected_id = None  # The instance to boot up
    for i, e in enumerate(instances):
        print_instance(e)

    # 1. Choose or create instance
    # ----------------------------------------
    while len(instances) >= 1:
        try:
            s = input("Choose an instance or type 'n' to create a new one: ")
            if s == 'n':
                break;
            selected_id = instances[int(s) - 1]['id']
            break
        except:
            print("Invalid choice")

    if selected_id is None:
        # Create new instance
        # ----------------------------------------
        while True:
            offers = fetch_offers()

            # Sort the offers by price descending
            offers = sorted(offers, key=lambda e: float(e['price']), reverse=True)

            # Print the list of machines
            for i, e in enumerate(offers):
                print_offer(e)

            # Ask user to choose a machine, keep asking until valid choice
            try:
                choice = input("Enter the number of the machine you want to use: ")
                choice = int(choice)
                if 1 <= choice <= len(offers):
                    print_offer(e)
                    print("nice choice!")
                    break
            except:
                print("Invalid choice. Try again, or type r to refresh the list and see again.")

        # Create the machine
        # Example command: ./vast create instance 36842 --image vastai/tensorflow --disk 32
        selected_id = offers[choice - 1]['id']
        out = subprocess.check_output(['python3', vastpath, 'create', 'instance', selected_id, '--image', 'pytorch/pytorch', '--disk', '32']).decode('utf-8')
        if 'Started.' not in out:
            print("Failed to create instance:")
            print(out)
            return

        time.sleep(3)

        print(f"Successfully created instance {choice}!")

    def wait_for_instance(id):
        printed_loading = False
        ins = None
        while ins is None or ins['status'] != 'running':
            all_ins = [i for i in fetch_instances() if i['id'] == id]
            if len(all_ins) > 0:
                ins = all_ins[0]

                if ins is not None and ins['status'] == 'running':
                    return ins

                if ins is not None and ins['status'] != 'running':
                    if not printed_loading:
                        print(f"Waiting for {id} to finish loading...")
                        printed_loading = True

            time.sleep(3)

    # 2. Wait for instance to be ready
    # ----------------------------------------
    time.sleep(3)
    instance = wait_for_instance(selected_id)

    # 3. Connections
    # ----------------------------------------
    username = 'root'
    ip = instance['sshaddr']
    port = instance['sshport']

    import paramiko
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=int(port), username='root')

    print(f"ssh -p {port} root@{ip}")
    print(f"kitty +kitten ssh -p {port} root@{ip}")
    print("")

    if user_conf.vastai_sshfs:
        # Use sshfs to mount the machine
        d = Path(user_conf.vastai_sshfs_path).expanduser() / 'discore_deploy'
        d.mkdir(parents=True, exist_ok=True)
        os.system(f"umount {d}")
        os.system(f"sshfs root@{ip}:/workspace -p {port} {d}")

    # Open a shell
    if args.shell:
        # Start a ssh shell for the user
        channel = ssh.invoke_shell()
        interactive.interactive_shell(channel)

    # Deploy like in local
    src = paths.root
    dst = Path("/workspace/discore_deploy")
    sshexec(ssh, "ls /workspace/")

    # Deployment steps
    cmds = get_deploy_commands(dst.as_posix())
    for cmd in cmds:
        sshexec(ssh, ' '.join(cmd))

    if user_conf.vastai_sshfs:
        d = Path(user_conf.vastai_sshfs_path).expanduser()
        open_in_explorer(d)

    from src_core.ssh.sftpclient import SFTPClient
    sftp = SFTPClient.from_transport(ssh.get_transport())
    for file in deploy_copy:
        src_file = src / file
        dst_file = dst / file
        if src_file.is_dir():
            sftp.mkdir(dst_file.as_posix(), ignore_existing=True)
            sftp.put_dir(src_file.as_posix(), dst_file.as_posix())
        else:
            print(f"Uploading {src_file.as_posix()} to {dst_file.as_posix()}")
            sftp.put(src_file.as_posix(), dst_file.as_posix())
    sftp.close()

    sshexec(ssh, f"chmod +x {dst / 'discore.py'}", True)
    sshexec(ssh, f"{dst / 'discore.py'} --upgrade", True)

    # Start a ssh shell for the user
    # channel = ssh.invoke_shell()
    # interactive.interactive_shell(channel)


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
