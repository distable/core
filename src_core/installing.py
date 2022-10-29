import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import user_conf
from src_core import paths

git = os.environ.get('GIT', "git")
python = sys.executable

current_parent = ""


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def pipargs(args, desc=None):
    return run(f'"{python}" -m pip {args} --prefer-binary', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def pipinstall(package, desc=None):
    return pipargs(f"install {package}", desc)


def pipreqs(file):
    file = Path(file)
    return pipargs(f"install -r {file.as_posix()}", "requirements for Web UI")


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def run(command, desc=None, errdesc=None):
    if user_conf.print_more:
        print(f"  >> {command}")
    elif desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        # raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def gitclone(giturl, hash='master', repodir=None, name=None):
    # TODO clone into temporary dir and move if successful

    if name is None:
        name = Path(giturl).stem.replace('-', '_')

    if repodir is None:
        repodir = Path(paths.plug_repos) / current_parent / name

    if not repodir.exists():
        run(f'"{git}" clone {giturl} {Path(repodir)}')
    else:
        current_hash = run(f'"{git}" -C {repodir} rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {hash}").strip()
        if current_hash != hash:
            run(f'"{git}" -C {repodir} fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
            # print(giturl, clonedir, name, commithash)

            if hash is not None and hash != 'master':
                run(f'"{git}" -C {repodir} checkout {hash}', f"Checking out commit for {name} with hash: {hash}...", f"Couldn't checkout {name}'s hash: {hash}")

    sys.path.append(repodir.as_posix())

    def mvfiles(src_path: str, dest_path: str, ext_filter: str = None):
        """
        Move some files from a source directory to a destination directory.
        Args:
            src_path: The source directory.
            dest_path: The destination directory.
            ext_filter: A file extension filter. Only files with this extension will be moved.

        Returns: None
        """
        try:
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            if os.path.exists(src_path):
                for file in os.listdir(src_path):
                    fullpath = os.path.join(src_path, file)
                    if os.path.isfile(fullpath):
                        if ext_filter is not None:
                            if ext_filter not in file:
                                continue
                        print(f"Moving {file} from {src_path} to {dest_path}.")
                        try:
                            shutil.move(fullpath, dest_path)
                        except:
                            pass
                if len(os.listdir(src_path)) == 0:
                    print(f"Removing empty folder: {src_path}")
                    shutil.rmtree(src_path, True)
        except:
            pass
