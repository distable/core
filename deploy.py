import os
import subprocess
import sys
import time
from pathlib import Path

from paramiko.ssh_exception import NoValidConnectionsError, SSHException
from yachalk import chalk

import jargs
from jargs import args, argv
from src_core.classes import paths
from src_core.lib.corelib import open_in_explorer, shlexrun, shlexrun_err
from src_core.rendering import renderer

# User files/directories to copy at the start of a deployment
deploy_rsync = [('requirements-vastai.txt', 'requirements.txt'),
                'discore.py',
                'deploy.py',
                'jargs.py',
                paths.userconf_name,
                paths.scripts_name,
                paths.src_core_name,
                paths.src_plugins_name,
                # paths.plug_repos_name
                ]

deploy_put = [paths.plug_res_name]
deploy_put = []

excludes_rsync_download = [
    "video.mp4",
    "video__*.mp4",
    "script.py",
    "*.npy",
]

excludes_rsync_upload = [
    "video.mp4",
    "video__*.mp4",
    # "*.npy",
]


# Commands to run in order to setup a deployment
def get_deploy_commands(clonepath):
    return [
        ['git', 'clone', '--recursive', 'https://github.com/distable/core', clonepath],
        ['git', '-C', clonepath, 'submodule', 'update', '--init', '--recursive'],
    ]


def deploy_local():
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

    for file in deploy_rsync:
        if isinstance(file, tuple):
            shutil.copyfile(src / file[0], dst / file[1])
        if isinstance(file, str):
            shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])

def print_header(string):
    print("")
    print("----------------------------------------")
    print(chalk.green(string))
    print("----------------------------------------")
    print("")

def deploy_vastai():
    """
    Deploy onto cloud.
    """
    import paramiko
    import userconf
    import json
    from src_core.deploy.sftpclient import sshexec

    session = jargs.get_discore_session()

    # 1. List the available machines with vastai api
    # 2. Prompt the user to choose one
    # 3. Connect to it with SSH
    # 4. Git clone the core repository
    # 5. Upload our userconf
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
        import userconf
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'search', 'offers', args.vastai_search or userconf.vastai_default_search]).decode('utf-8')
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

    def fetch_balance():
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'show', 'invoices']).decode('utf-8')
        # Example output (only the last line)
        # Current:  {'charges': 0, 'service_fee': 0, 'total': 0, 'credit': 6.176303554744997}

        lines = out.splitlines()
        s = lines[-1]
        s = s.replace('Current:  ', '')
        s = s.replace("'", '"')
        o = json.loads(s)
        return float(o['credit'])


    def fetch_instances():
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'show', 'instances']).decode('utf-8')
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

    def print_offer(e, i):
        print(chalk.green(f'{i + 1} - {e["model"]} - {e["num"]} - {e["dlp"]} - {e["netdown"]} Mbps - {e["price"]} $/hr - {e["dlpprice"]} DLP/HR'))

    def print_instance(e):
        print(chalk.green_bright(f'{i + 1} - {e["model"]} - {e["num"]} - {e["netdown"]} Mbps -  {e["price"]} $/hr - {e["status"]}'))

    print("Deployed instances:")
    instances = fetch_instances()
    selected_id = None  # The instance to boot up
    for i, e in enumerate(instances):
        print_instance(e)
    print("")

    # 1. Choose or create instance
    # ----------------------------------------
    if len(instances) >= 1:
        selected_id = instances[0]['id']
    # while len(instances) >= 1:
    #     try:
    #         s = input("Choose an instance or type 'n' to create a new one: ")
    #         if s == 'n':
    #             break;
    #         selected_id = instances[int(s) - 1]['id']
    #         break
    #     except:
    #         print("Invalid choice")

    if selected_id is None:
        # Create new instance
        # ----------------------------------------
        while True:
            offers = fetch_offers()

            # Sort the offers by price descending
            offers = sorted(offers, key=lambda e: float(e['price']), reverse=True)

            # Print the list of machines
            for i, e in enumerate(offers):
                print_offer(e, i)

            # Ask user to choose a machine, keep asking until valid choice
            print("")
            try:
                choice = input("Enter the number of the machine you want to use: ")
                choice = int(choice)
                if 1 <= choice <= len(offers):
                    print_offer(offers[choice - 1], choice - 1)
                    print()
                    break
            except:
                print("Invalid choice. Try again, or type r to refresh the list and see again.")

        # Create the machine
        # Example command: ./vast create instance 36842 --image vastai/tensorflow --disk 32
        selected_id = offers[choice - 1]['id']
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'create', 'instance', selected_id, '--image', 'pytorch/pytorch', '--disk', '32']).decode('utf-8')
        if 'Started.' not in out:
            print("Failed to create instance:")
            print(out)
            return

        time.sleep(3)

        new_instances = fetch_instances()
        # Diff between old and new instances
        new_instances = [e for e in new_instances if e['id'] not in [e['id'] for e in instances]]

        if len(new_instances) != 1:
            print("Failed to create instance, couldn't find the new instance by diffing.")
            return

        selected_id = new_instances[0]['id']
        print(f"Successfully created instance {selected_id}!")

    def wait_for_instance(id):
        printed_loading = False
        ins = None
        while ins is None or ins['status'] != 'running':
            all_ins = [i for i in fetch_instances() if i['id'] == id]
            if len(all_ins) > 0:
                ins = all_ins[0]

                status = ins['status']
                if ins is not None and status == 'running':
                    return ins

                if ins is not None and status != 'running':
                    if not printed_loading:
                        print("")
                        print(f"Waiting for {id} to finish loading (status={status})...")
                        printed_loading = True

            time.sleep(3)

        time.sleep(3)

    # 2. Wait for instance to be ready
    # ----------------------------------------
    instance = wait_for_instance(selected_id)

    # 3. Connections
    # ----------------------------------------
    user = 'root'
    ip = instance['sshaddr']
    port = instance['sshport']

    ssh_cmd = f"ssh -p {port} {user}@{ip}"
    kitty_cmd = f"kitty +kitten {ssh_cmd}"

    print("")
    print('----------------------------------------')
    print(chalk.green(f"Establishing connections {user}@{ip}:{port}..."))
    print(ssh_cmd)
    print(kitty_cmd)
    print('----------------------------------------')
    print("")

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    while True:
        try:
            # This can fail when the instance just launched
            ssh.connect(ip, port=int(port), username='root')
            break
        except NoValidConnectionsError as e:
            print(f"Failed to connect ({e}), retrying...")
            time.sleep(3)
        except SSHException as e:
            os.system(kitty_cmd)

    from src_core.deploy.sftpclient import SFTPClient
    sftp = SFTPClient.from_transport(ssh.get_transport())
    sftp.max_size = 10 * 1024 * 1024
    sftp.urls = []
    if hasattr(userconf, 'deploy_urls'):
        sftp.urls = userconf.deploy_urls
    sftp.ssh = ssh
    sftp.enable_urls = not args.vastai_no_download
    sftp.ip = ip
    sftp.port = port


    def sshexec(ssh, cm, cwd=None, require_output=False):
        cm = cm.replace("'", '"')
        if cwd is not None:
            cm = f"cd {cwd}; {cm}"
        proc = subprocess.Popen(f"{ssh_cmd} '{cm}'", shell=True)
        ret = proc.wait()
        return ret

    def file_exists(ssh, path):
        ret = sshexec(ssh, f"stat '{path}'", require_output=True)
        return ret == 0
        # print(out)
        # return 'No such file or directory' not in out

    # sshexec(ssh, f'rm -rf ')

    src = paths.root
    dst = Path("/workspace/discore_deploy")
    is_fresh_install = not (file_exists(ssh, dst) and jargs.is_vastai_continue)

    # Mount
    # ----------------------------------------
    mountdir = None
    if userconf.vastai_sshfs:
        print_header("Mounting with sshfs...")
        # Use sshfs to mount the machine
        mountdir = Path(userconf.vastai_sshfs_path).expanduser() / 'discore_deploy'
        mountdir.mkdir(parents=True, exist_ok=True)

        not_mounted = subprocess.Popen(f"mountpoint -q {mountdir}", shell=True).wait()
        if not_mounted > 0:
            os.system(f"sshfs root@{ip}:/workspace -p {port} {mountdir}")

    # rm -rf existing & Clone
    # ----------------------------------------
    if is_fresh_install:
        if file_exists(ssh, dst):
            print(chalk.red("Removing old deploy..."))
            time.sleep(3)
            sshexec(ssh, f"rm -rf {dst}")

        print_header("Cloning")
        cmds = get_deploy_commands(dst.as_posix())
        for cmd in cmds:
            sshexec(ssh, ' '.join(cmd))

    # if is_fresh_install:
    #     mountdir = Path(userconf.vastai_sshfs_path).expanduser()
    #     open_in_explorer(mountdir)

    # Install deps
    # ----------------------------------------

    if is_fresh_install:
        print_header("Installing system libraries...")
        sshexec(ssh, f"apt-get install python3-venv -y")
        sshexec(ssh, f"apt-get install libgl1 -y")
        sshexec(ssh, f"apt-get install zip -y")
        sshexec(ssh, f"apt-get install ffmpeg -y")
        sshexec(ssh, f"chmod +x {dst / 'discore.py'}")
        # sshexec(ssh, f"rm -rf {dst / 'venv'}")

    # Copy files
    # ----------------------------------------
    # if is_fresh_install or args.vastai_copy:
    print_header("File copies...")
    for file in deploy_rsync:
        if isinstance(file, str):
            sftp.put_any(src / file, dst / file, rsync_includes=[f'*{v}' for v in paths.text_exts])
        if isinstance(file, tuple):
            sftp.put_any(src / file[0], dst / file[1])

    # Copy res
    # ----------------------------------------
    # This is slow and usually only needed for the first time
    if is_fresh_install or args.vastai_copy:
        for file in deploy_put:
            if isinstance(file, str):
                sftp.put_any(src / file, dst / file, forbid_rsync=True)
            if isinstance(file, tuple):
                sftp.put_any(src / file[0], dst / file[1], forbid_rsync=True)

        sftp.close()

    # --shell
    # ----------------------------------------
    if args.shell:
        import interactive
        print_header("user --shell")
        # Start a ssh shell for the user
        channel = ssh.invoke_shell()
        interactive.interactive_shell(channel)

    # pip & plugin install
    # ----------------------------------------
    if is_fresh_install or args.vastai_upgrade:
        print_header("Discore pip refresh")
        launch_cmd = f"{ssh_cmd} 'cd /workspace/discore_deploy/; /opt/conda/bin/python3 {dst / 'discore.py'} --upgrade --no_venv'"
        os.system(launch_cmd)

        print_header("Discore plugin install")
        launch_cmd = f"{ssh_cmd} 'cd /workspace/discore_deploy/; /opt/conda/bin/python3 {dst / 'discore.py'} --install --dry'"
        os.system(launch_cmd)

    # Transfer the session state
    if not args.vastai_continue_quick:
        if session is not None and session.dirpath.exists():
            print_header(f"Copying work session '{session.dirpath.stem}'")

            dst_session = dst / 'sessions' / session.dirpath.stem
            sftp.put_any(session.dirpath, dst_session, forbid_recursive=True, rsync_excludes=excludes_rsync_upload)

            # Open in nvim
            # if session.res("script.py").exists() and userconf.vastai_sshfs:
            #     d = mountdir / 'discore_deploy' / paths.sessions_name / session.dirpath.stem / 'script.py'
            #     subprocess.Popen(f"kitty nvim {d}", shell=True)

            # sftp.mkdir(str(dst_session))
            # # Iterate files in dirpath
            # for file in session.dirpath.iterdir():
            #     # If the stem is not an int
            #     if not file.stem.isnumeric():
            #         sftp.put_any(str(file), str(dst_session / file.name), forbid_rsync=True)

    continue_work = True
    import threading
    def vastai_job():
        launch_cmd = f"{ssh_cmd} 'cd /workspace/discore_deploy/; /opt/conda/bin/python3 {dst / 'discore.py'}"

        oargs = argv
        safe_list_remove(oargs, '--vastai')
        safe_list_remove(oargs, '--vai')
        safe_list_remove(oargs, '--vastai_continue')
        safe_list_remove(oargs, '--vastai_continue_quick')
        safe_list_remove(oargs, '--vaic')
        launch_cmd += f' {" ".join(oargs)}'

        launch_cmd += f' --remote'
        launch_cmd += f' --no_venv'
        launch_cmd += f' --cli'
        launch_cmd += f' --unsafe'
        if is_fresh_install:
            launch_cmd += f' --install'

        launch_cmd += "'"

        print_header("Launching discore for work ...")
        print("")
        print(f'> {launch_cmd}')
        os.system(launch_cmd)

        nonlocal continue_work
        continue_work = False
        renderer.request_stop = True


    def balance_job():
        """
        Notify the user how much credit is left, every 0.25$
        """
        from desktop_notifier import DesktopNotifier

        threshold = 0.25

        notifier = DesktopNotifier()
        last_balance = None
        while continue_work:
            balance = fetch_balance()
            if last_balance is None or balance - last_balance > threshold:
                last_balance = balance
                notifier.send_sync(title='Vast.ai balance', message=f'{balance:.02f}$')

            time.sleep(5)

    def detect_changes_job():
        """
        Detect changes to the code (in src/scripts) and copy them up to the server (to dst/scripts)
        """
        from src_core.deploy.watch import Watcher

        changed_files = []

        def execute(f):
            nonlocal changed_files
            changed_files.append(f)

        watch = Watcher([paths.scripts, session.dirpath / 'script.py'], [execute])
        elapsed = 0
        while continue_work:
            time.sleep(0.5)
            elapsed += 0.5
            watch.monitor_once()

            for f in changed_files:
                f = Path(f)
                print(chalk.blue_bright("Changed", f.relative_to(src)))
                relative = Path(f).relative_to(src)

                src2 = src / relative
                dst2 = dst / relative
                os.system(f"rsync -avz -e 'ssh -p {port}' {src2} root@{ip}:{dst2}")

            changed_files.clear()

            # Download the session every 5 second
            if elapsed > 5:
                src2 = dst / paths.sessions_name / session.name
                dst2 = src / paths.sessions_name

                # Exclude video.mp4 and video__*.mp4
                cmd = f"rsync -az -e 'ssh -p {port}' root@{ip}:{src2} {dst2}"

                for fname in excludes_rsync_download:
                    cmd += f" --exclude '{fname}'"

                os.system(cmd)
                elapsed = 0

    # TODO readonly renderer job
    def renderer_job():
        """
        Render the video in the background
        """
        # renderer.is_dev = True
        renderer.is_readonly = True
        renderer.is_dev= True
        renderer.unsafe = False
        renderer.init(main_thread=True)
        renderer.loop()

    t1 = threading.Thread(target=vastai_job)
    t2 = threading.Thread(target=balance_job)
    t3 = threading.Thread(target=detect_changes_job)
    # t4 = threading.Thread(target=renderer_job)

    t1.start()
    t2.start()
    t3.start()
    # t4.start()
    # vastai_job()
    renderer_job()
    t1.join()
    t2.join()
    t3.join()
    # t4.join()

    print(f"Remaining balance: {fetch_balance():.02f}$")

    if args.vastai_stop:
        print(f"Stopping Vast.ai instance {selected_id} in 5 seconds ...")
        time.sleep(5)
        subprocess.check_output(['python3', vastpath.as_posix(), 'stop', str(selected_id)]).decode('utf-8')
        print("All done!")


    # interactive.interactive_shell(channel)


def safe_list_remove(l, value):
    if not l: return
    try:
        l.remove(value)
    except:
        pass
