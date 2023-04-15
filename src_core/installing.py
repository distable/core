import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src_core.classes import paths

git = os.environ.get('GIT', "git")
python = sys.executable

default_basedir = ""
skip_installations = False


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
    return run(f'"{python}" -m pip {args} --prefer-binary', err=f"Couldn't install {desc}")


def pipinstall(package, desc=None):
    return pipargs(f"install {package}", desc)


def pipreqs(file):
    if not skip_installations:
        file = Path(file)
        return pipargs(f"install -r {file.as_posix()}", "requirements for Web UI")
    return ""


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')

def run(command, log: bool | str | None = False, err=None):
    if log:
        if isinstance(log, str):
            print(log)
        elif log is True:
            print(f"  >> {command}")

    result = subprocess.run(command, shell=True) # stdout=subprocess.PIPE, stderr=subprocess.PIPE,

#     if result.returncode != 0:
#         message = f"""{err or 'Error running command'}.
# Command: {command}
# Error code: {result.returncode}
# stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
# stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
# """
#         print(message)
#         # raise RuntimeError(message)

    if result.stdout:
        return result.stdout.decode(encoding="utf8", errors="ignore").strip()
    else:
        return None


def gitclone(giturl, hash='master', into_dir=None, name=None):
    # TODO clone into temporary dir and move if successful

    if name is None:
        name = Path(giturl).stem.replace('-', '_')

    if into_dir is None:
        into_dir = default_basedir

    clone_dir = Path(into_dir) / name

    if not skip_installations:
        if not clone_dir.exists():
            run(f'"{git}" clone {giturl} {Path(clone_dir)}')
        else:
            current_hash = run(f'"{git}" -C {clone_dir} rev-parse HEAD', err=f"Couldn't determine {name}'s hash: {hash}")
            if current_hash != hash:
                run(f'"{git}" -C {clone_dir} fetch', err=f"Couldn't fetch {name}")
                # print(giturl, clonedir, name, commithash)

                if hash is not None and hash != 'master':
                    run(f'"{git}" -C {clone_dir} checkout {hash}', err=f"Couldn't checkout {name}'s hash: {hash}")

    sys.path.append(str(clone_dir.parent))
    sys.path.append(str(clone_dir))

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


def print_info():
    from src_core.installing import git
    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"
    print(f"Python: {sys.version}")
    print(f"Revision: {commit}")


def open_explorer(directory):
    if sys.platform == 'win32':
        subprocess.Popen(['start', directory], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    elif sys.platform == 'darwin':
        subprocess.Popen(['open', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    else:
        try:
            subprocess.Popen(['xdg-open', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            pass
            # er, think of something else to try
            # xdg-open *should* be supported by recent Gnome, KDE, Xfce


def wget(file, url):
    if not file.exists():
        run(f'wget {url} --output-document {file.as_posix()}', err=f"Couldn't download {file.name}")