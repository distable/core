import re
from datetime import datetime
from pathlib import Path

from PIL.Image import Image

from src_core import paths, plugins, printlib
from src_core.PipeData import PipeData

mprint = printlib.make_print("session")
mprinterr = printlib.make_printerr("session")


# region Library
class Session:
    def __init__(self, name=None, path: Path | str = None):
        self.data = PipeData()
        self.jobs = []

        if name is not None:
            self.name = name
            self.path = paths.sessions / name
        elif path is not None:
            self.path = Path(path)
            self.name = Path(path).stem
        else:
            self.valid = False
            mprinterr("Cannot create session! No name or path given!")
            return

        if not self.path.exists():
            mprint("New session:", self.name)
        else:
            self.load_if_exists()

    def create_if_missing(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def load_if_exists(self):
        if self.path.exists():
            mprint(f"Loading session: {self.name}")
            # TODO load a session metadata file

    def save_next(self, dat):
        path = self.path / str(get_next_leadnum() + 1)
        self.save(dat, path)

    def save(self, dat, path):
        if isinstance(dat, list):
            for d in dat:
                self.save_next(d)
        elif isinstance(dat, Image):
            self.create_if_missing()
            dat.save(path.with_suffix(".png"))
        elif dat is None:
            pass
        else:
            mprinterr("Cannot save! unknown data type:", type(dat))

    def add_job(self, j):
        self.jobs.append(j)

    def rem_job(self, j):
        self.jobs.remove(j)


def get_next_leadnum(iterator=None, separator='_'):
    """
    Find the largest 'leading number' in the directory names and return it
    e.g.:
    23_session
    24_session
    28_session
    23_session

    return value is 28
    """
    iterator = iterator if iterator is not None else paths.sessions.iterdir()

    biggest = 0
    for path in iterator:
        if path.is_dir():
            match = re.match(r"^(\d+)" + separator, path.name)
            if match is not None:
                num = int(match.group(1))
                if match:
                    biggest = max(biggest, num)

    return biggest


def format_session_id(name, num=None):
    if num is None:
        num = get_next_leadnum()

    return f"{num:0>3}_{name}"


# endregion

# region API
def new_timestamp():
    """
    Returns: A new session which is timestamped, e.g.
    """
    return Session(format_session_id(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


def run(params=None, cmd=None, **kwargs):
    """
    Run a job in the session's context, meaning the output JobState data will be saved to disk
    """
    ret = plugins.run(params, cmd, print=mprint, **kwargs)
    current.save_next(ret)


# endregion


current = new_timestamp()
