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


