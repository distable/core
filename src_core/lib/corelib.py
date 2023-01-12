import os
from pathlib import Path


def to_dict(f):
    return dict((key, value) for key, value in f.__dict__.items() if not callable(value) and not key.startswith('__'))


def open_in_explorer(path):
    path = Path(path).as_posix()
    if os.name == 'nt':
        os.startfile(path)
    elif os.name == 'linux' or os.name == 'posix':
        os.system(f'xdg-open "{path}"')
    else:
        raise Exception(f"open_in_explorer: Unsupported OS '{os.name}' ")


