import os
import subprocess
from colorsys import hsv_to_rgb
from pathlib import Path
import time

from src_core.classes.printlib import trace


rgb_to_hex = lambda tuple: f"#{int(tuple[0] * 255):02x}{int(tuple[1] * 255):02x}{int(tuple[2] * 255):02x}"


def generate_colors(n, s=0.825, v=0.915):
    golden_ratio_conjugate = 0.618033988749895
    h = 0
    ret = []
    for i in range(n):
        h += golden_ratio_conjugate
        ret.append(rgb_to_hex(hsv_to_rgb(h % 1, s, v)))

    return ret


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


def shlexrun(cmd, print_cmd=True, shell=False, **kwargs):
    import shlex
    import subprocess
    if print_cmd:
        from yachalk import chalk
        print(chalk.grey(f"> {cmd}"))
    proc = subprocess.Popen(shlex.split(cmd), shell=shell, **kwargs)
    proc.wait()
    return proc


def shlexrun_err(cm, print_cmd=True):
    proc = shlexrun(cm, print_cmd=print_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    return proc.stdout.decode('utf-8')


def shlexproc(cmd, **kwargs):
    import shlex
    import subprocess
    print(cmd)
    return subprocess.Popen(shlex.split(cmd), **kwargs)


def shlexproc_err(cm):
    proc = shlexproc(cm, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    return proc.stdout.decode('utf-8')

def invoke_safe(func, *kargs, failsleep=0.0, unsafe=False, **kwargs):
    if isinstance(func, list):
        for f in func:
            if unsafe:
                f(*kargs, **kwargs)
                continue

            if not invoke_safe(f, *kargs, failsleep=failsleep, **kwargs):
                return False

        return True
    else:
        if unsafe:
            func(*kargs, **kwargs)
            return True

        try:
            with trace(f"safe_call({func.__name__ if hasattr(func, '__name__') else func.__class__.__name__})"):
                func(*kargs, **kwargs)
            return True
        except Exception as e:
            # Print the full stacktrace
            import traceback
            traceback.print_exc()
            print(e)
            time.sleep(failsleep)
            return False
