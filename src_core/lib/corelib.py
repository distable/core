import os
import subprocess
from pathlib import Path


def to_dict(f):
    return dict((key, value) for key, value in f.__dict__.items() if not callable(value) and not key.startswith('__'))


def open_in_explorer(path):
    path = Path(path).as_posix()
    if os.name == 'nt':
        os.startfile(path)
    elif os.name == 'linux' or os.name == 'posix':
        subprocess.Popen(f'xdg-open "{path}"', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    else:
        raise Exception(f"open_in_explorer: Unsupported OS '{os.name}' ")


def shlexrun(cmd, **kwargs):
    import shlex
    import subprocess
    print(cmd)
    return subprocess.run(shlex.split(cmd), **kwargs)


def shlexrun_err(cm):
    proc = shlexrun(cm, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    return proc.stdout.decode('utf-8')

def shlexproc(cmd, **kwargs):
    import shlex
    import subprocess
    print(cmd)
    return subprocess.Popen(shlex.split(cmd), **kwargs)

def shlexproc_err(cm):
    proc = shlexproc(cm, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    return proc.stdout.decode('utf-8')