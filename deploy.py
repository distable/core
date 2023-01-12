import os
import subprocess
import sys
import time
from pathlib import Path

from jargs import args, original_args
from src_core.classes import paths
from src_core.lib.corelib import open_in_explorer


# User files/directories to copy at the start of a deployment
deploy_copy = ['requirements.txt', 'discore.py', paths.userconf_name, paths.scripts_name, paths.plug_res_name]


# Commands to run in order to setup a deployment
def get_deploy_commands(clonepath):
    return [
        ['git', 'clone', '--recursive', 'https://github.com/distable/core', clonepath],
        ['git', '-C', clonepath, 'submodule', 'update', '--init', '--recursive'],
    ]


def deploy_vastai():
    """
    Deploy onto cloud.
    """
    import interactive
    import paramiko
    import user_conf
    from src_core.deploy.sftpclient import sshexec

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
        import user_conf
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'search', 'offers', args.vastai_search or user_conf.vastai_default_search]).decode('utf-8')
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

    def print_offer(e):
        print(chalk.green(f'{i + 1} - {e["model"]} - {e["num"]} - {e["dlp"]} - {e["price"]} $/hr - {e["dlpprice"]} DLP/HR'))

    def print_instance(e):
        print(chalk.green_bright(f'{i + 1} - {e["model"]} - {e["num"]} -  {e["price"]} $/hr - {e["status"]}'))

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
        out = subprocess.check_output(['python3', vastpath.as_posix(), 'create', 'instance', selected_id, '--image', 'pytorch/pytorch', '--disk', '32']).decode('utf-8')
        if 'Started.' not in out:
            print("Failed to create instance:")
            print(out)
            return

        time.sleep(3)

        print(f"Successfully created instance {selected_id}!")

    def wait_for_instance(id):
        printed_loading = False
        ins = None
        while ins is None or ins['status'] != 'running':
            all_ins = [i for i in fetch_instances() if i['id'] == id]
            print(all_ins)
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
    instance = wait_for_instance(selected_id)

    # 3. Connections
    # ----------------------------------------
    username = 'root'
    ip = instance['sshaddr']
    port = instance['sshport']
    print(chalk.green(f"Establishing connections root@{ip}:{port}..."))

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=int(port), username='root')

    from src_core.deploy.sftpclient import SFTPClient
    sftp = SFTPClient.from_transport(ssh.get_transport())
    sftp.max_size = 10 * 1024 * 1024
    sftp.urls = user_conf.deploy_urls
    sftp.ssh = ssh

    ssh_cmd = f"ssh -p {port} root@{ip}"
    kitty_cmd = f"kitty +kitten {ssh_cmd}"

    print(ssh_cmd)
    print(kitty_cmd)
    print("")

    # Print ls
    sshexec(ssh, "ls /workspace/")

    if user_conf.vastai_sshfs:
        print(chalk.green("Mounting with sshfs..."))
        # Use sshfs to mount the machine
        d = Path(user_conf.vastai_sshfs_path).expanduser() / 'discore_deploy'
        d.mkdir(parents=True, exist_ok=True)
        os.system(f"umount {d}")
        os.system(f"sshfs root@{ip}:/workspace -p {port} {d}")

    # ----------------------------------------
    src = paths.root
    dst = Path("/workspace/discore_deploy")
    if args.vastai_recreate:
        print(chalk.green("--vastai_recreate"))
        sshexec(ssh, f"rm -rf {dst}")

    repo_existed = sftp.exists(dst)
    print('repo_existed', repo_existed)

    # Deployment steps
    # ----------------------------------------
    print(chalk.green("Deployment commands..."))
    cmds = get_deploy_commands(dst.as_posix())
    for cmd in cmds:
        sshexec(ssh, ' '.join(cmd))

    # ----------------------------------------
    print(chalk.green("VastAI setup commands..."))
    sshexec(ssh, f"apt-get install python3-venv -y")
    sshexec(ssh, f"apt-get install libgl1 -y")
    sshexec(ssh, f"chmod +x {dst / 'discore.py'}")
    # sshexec(ssh, f"rm -rf {dst / 'venv'}")

    # ----------------------------------------
    print(chalk.green("Copying user files..."))
    if user_conf.vastai_sshfs and not repo_existed:
        d = Path(user_conf.vastai_sshfs_path).expanduser()
        open_in_explorer(d)

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


    # Open a shell
    if args.shell:
        print(chalk.green("--shell"))
        # Start a ssh shell for the user
        channel = ssh.invoke_shell()
        interactive.interactive_shell(channel)

    # Start a ssh shell for the user
    launch_cmd = f"{kitty_cmd} 'cd /workspace/discore_deploy/; /opt/conda/bin/python3 {dst / 'discore.py'}"
    oargs = original_args
    oargs.remove('--vastai')
    launch_cmd += f' {" ".join(oargs)}'
    launch_cmd += f' --upgrade --no_venv'
    launch_cmd += "'"

    print(launch_cmd)
    os.system(launch_cmd)

    # interactive.interactive_shell(channel)


def input_bool(string):
    while True:
        yes = input(string).lower() in ['yes', 'y', 'true', 't', '1']
        no = input(string).lower() in ['no', 'n', 'false', 'f', '0']
        if yes: return True
        if no: return False

        print("Please respond with 'y' or 'n'")


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

    for file in deploy_copy:
        shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])
